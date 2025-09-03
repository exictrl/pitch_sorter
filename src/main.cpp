#include <sndfile.h>

#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct FilePitch {
    fs::path path;
    double   freq_hz = 0.0;
    bool     ok = false;
};

static bool is_wav_file(const fs::path& p) {
    if (!p.has_extension()) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    return ext == ".wav";
}

static std::vector<float> interleaved_to_mono_avg(const std::vector<float>& in, int channels) {
    if (channels <= 1) return in;
    std::vector<float> mono(in.size() / static_cast<size_t>(channels));
    const size_t frames = mono.size();
    for (size_t i = 0; i < frames; ++i) {
        double sum = 0.0;
        for (int c = 0; c < channels; ++c) sum += in[i*channels + static_cast<size_t>(c)];
        mono[i] = static_cast<float>(sum / channels);
    }
    return mono;
}

static void normalize_peak(std::vector<float>& x, float target = 0.95f) {
    float peak = 0.f;
    for (float v : x) peak = std::max(peak, std::abs(v));
    if (peak > 0.f) {
        const float g = target / peak;
        for (auto& v : x) v *= g;
    }
}

static std::vector<float> resample_linear(const std::vector<float>& in, int src_sr, int dst_sr) {
    if (src_sr == dst_sr || in.empty()) return in;
    const double ratio = static_cast<double>(dst_sr) / static_cast<double>(src_sr);
    const size_t out_len = static_cast<size_t>(std::llround(in.size() * ratio));
    if (out_len < 2) return in;
    std::vector<float> out(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        const double pos = static_cast<double>(i) / ratio;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        if (idx + 1 >= in.size()) {
            out[i] = in.back();
        } else {
            out[i] = static_cast<float>((1.0 - frac) * in[idx] + frac * in[idx + 1]);
        }
    }
    return out;
}

// Возвращает оценку частоты (Гц) по автокорреляции или std::nullopt, если неуверенно.
static std::optional<double> detect_pitch_acf(const std::vector<float>& x_full, int sr) {
    if (x_full.size() < static_cast<size_t>(sr / 10)) return std::nullopt; // хотя бы 0.2с

    // Берём "самый энергичный" 1-секундный сегмент, чтобы избегать тишины.
    const size_t win = static_cast<size_t>(sr);
    size_t best_start = 0;
    double best_energy = -1.0;
    if (x_full.size() <= win) {
        best_start = 0;
    } else {
        const size_t hop = win / 2;
        for (size_t s = 0; s + win <= x_full.size(); s += hop) {
            double e = 0.0;
            for (size_t i = 0; i < win; ++i) {
                double v = x_full[s + i];
                e += v * v;
            }
            if (e > best_energy) { best_energy = e; best_start = s; }
        }
    }

    const size_t N = std::min(win, x_full.size() - best_start);
    if (N < 600) return std::nullopt;

    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = x_full[best_start + i];

    // Убираем DC-смещение и применяем Хэнн.
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(N);
    const double two_pi = 6.283185307179586;
    for (size_t i = 0; i < N; ++i) {
        double w = 0.5 * (1.0 - std::cos(two_pi * static_cast<double>(i) / static_cast<double>(N - 1)));
        x[i] = (x[i] - mean) * w;
    }

    // Диапазон лагов для 80–400 Гц.
    int minLag = static_cast<int>(std::floor(static_cast<double>(sr) / 600.0));
    int maxLag = static_cast<int>(std::ceil (static_cast<double>(sr) /  50.0));
    if (minLag < 1) minLag = 1;
    if (maxLag >= static_cast<int>(N) - 2) maxLag = static_cast<int>(N) - 3;
    if (maxLag <= minLag + 2) return std::nullopt;

    // ACF[0] — энергия.
    double r0 = 0.0;
    for (size_t i = 0; i < N; ++i) r0 += x[i] * x[i];
    if (r0 <= 1e-12) return std::nullopt;

    std::vector<double> acf(maxLag + 1, 0.0);
    for (int lag = minLag; lag <= maxLag; ++lag) {
        double s = 0.0;
        const size_t M = N - static_cast<size_t>(lag);
        for (size_t i = 0; i < M; ++i) s += x[i] * x[i + static_cast<size_t>(lag)];
        acf[lag] = s / r0; // нормируем
    }

    // Ищем максимум в диапазоне и уточняем параболой.
    int bestLag = minLag;
    double bestVal = acf[minLag];
    for (int lag = minLag + 1; lag <= maxLag - 1; ++lag) {
        if (acf[lag] > bestVal) {
            bestVal = acf[lag];
            bestLag = lag;
        }
    }

    // Порог корреляции (эмпирически): если слабая — считать неуверенным.
    if (bestVal < 0.12) return std::nullopt;

    // Параболическая интерполяция вокруг пика (lag-1, lag, lag+1)
    double refined = static_cast<double>(bestLag);
    {
        const double y_m1 = acf[bestLag - 1];
        const double y_0  = acf[bestLag];
        const double y_p1 = acf[bestLag + 1];
        const double denom = (y_m1 - 2.0 * y_0 + y_p1);
        if (std::abs(denom) > 1e-12) {
            const double delta = 0.5 * (y_m1 - y_p1) / denom; // в пределах [-1, 1]
            refined += std::clamp(delta, -1.0, 1.0);
        }
    }

    const double freq = static_cast<double>(sr) / refined;
    if (freq < 50.0 || freq > 600.0) return std::nullopt; // защита от артефактов
    return freq;
}

static void print_usage(const char* exe) {
    std::cout << "Directory:\n  " << exe << " <input_directory>\n"
              << "Flags:\n  -h, --help   Short help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 0; }
    const std::string arg1 = argv[1];
    if (arg1 == "-h" || arg1 == "--help") { print_usage(argv[0]); return 0; }

    fs::path input = fs::u8path(arg1);
    if (!fs::exists(input) || !fs::is_directory(input)) {
        std::cerr << "Error: Directory does not exist or is not a directory: " << input.string() << "\n";
        return 1;
    }

    // Сканируем только файлы верхнего уровня (без рекурсии)
    std::vector<fs::path> wavs;
    for (const auto& de : fs::directory_iterator(input)) {
        if (!de.is_regular_file()) continue;
        if (is_wav_file(de.path())) wavs.push_back(de.path());
    }

    if (wavs.empty()) {
        std::cout << "WAV files not found in: " << input.string() << "\n";
        return 0;
    }

    std::vector<FilePitch> results;
    results.reserve(wavs.size());

    std::cout << "Processing files (" << wavs.size() << "):\n";

    for (const auto& p : wavs) {
        SF_INFO info{};
        SNDFILE* snd = sf_open(p.string().c_str(), SFM_READ, &info);
        if (!snd) {
            std::cerr << "  [Error] " << p.filename().string() << " — could not open.\n";
            results.push_back({p, 0.0, false});
            continue;
        }

        const sf_count_t frames_to_read = info.frames;
        std::vector<float> interleaved(static_cast<size_t>(frames_to_read) * static_cast<size_t>(info.channels));
        sf_count_t actually = sf_readf_float(snd, interleaved.data(), frames_to_read);
        sf_close(snd);
        interleaved.resize(static_cast<size_t>(actually) * static_cast<size_t>(info.channels));

        // В моно
        auto mono = interleaved_to_mono_avg(interleaved, info.channels);

        // Приводим к 44100 Гц (если нужно)
        int sr = info.samplerate;
        if (sr <= 0) sr = 44100;
        if (sr != 44100) {
            mono = resample_linear(mono, sr, 44100);
            sr = 44100;
        }

        // Нормализация уровня
        normalize_peak(mono, 0.95f);

        // Оценка основного тона
        auto freq_opt = detect_pitch_acf(mono, sr);
        double f = freq_opt.value_or(0.0);
        results.push_back({p, f, freq_opt.has_value()});

        if (freq_opt)
            std::cout << "  " << p.filename().string() << "  ->  " << std::lround(f) << " Hz\n";
        else
            std::cout << "  " << p.filename().string() << "  ->  [not defined]\n";
    }

    // Сортировка: сначала все с определённой частотой по возрастанию, затем "не определено"
    std::stable_sort(results.begin(), results.end(), [](const FilePitch& a, const FilePitch& b){
        if (a.ok != b.ok) return a.ok > b.ok; // ok сначала
        return a.freq_hz < b.freq_hz;
    });

    // Создаём выходную директорию
    fs::path out_dir = input / "sorted_output";
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Error creating directory: " << out_dir.string() << " (" << ec.message() << ")\n";
        return 2;
    }

    const int digits = std::max(3, static_cast<int>(std::to_string(results.size()).size()));
    size_t counter = 1;

    double minF = std::numeric_limits<double>::infinity();
    double maxF = 0.0;
    size_t ok_count = 0;

    for (const auto& r : results) {
        std::ostringstream num;
        num << std::setw(digits) << std::setfill('0') << counter++;

        std::string freqTag = r.ok ? (std::to_string(static_cast<int>(std::lround(r.freq_hz))) + "Hz")
                                   : std::string("NAHz");

        std::string original = r.path.filename().string();
        // гарантируем расширение .wav (оставим оригинальное имя целиком)
        //fs::path out_name = num.str() + "_" + freqTag + "_" + original;
        fs::path out_name = freqTag + "_" + original;
        fs::path dst = out_dir / out_name;

        std::error_code copy_ec;
        fs::copy_file(r.path, dst, fs::copy_options::overwrite_existing, copy_ec);
        if (copy_ec) {
            std::cerr << "  [Error copying] " << r.path.filename().string()
                      << " -> " << dst.filename().string() << " (" << copy_ec.message() << ")\n";
        }

        if (r.ok) {
            minF = std::min(minF, r.freq_hz);
            maxF = std::max(maxF, r.freq_hz);
            ++ok_count;
        }
    }

    // Summary
    std::cout << "\n- Summary -\n";
    std::cout << "Processed WAV files: " << results.size() << "\n";
    std::cout << "With defined frequency: " << ok_count << "\n";
    if (ok_count > 0) {
        std::cout << "Frequency range: " << std::lround(minF) << "-" << std::lround(maxF) << " Hz\n";
    }
    std::cout << "Output folder: " << out_dir.string() << "\n";

    return 0;
}

