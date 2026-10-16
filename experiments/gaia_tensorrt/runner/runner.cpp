// Standalone C++ runner for the serialized engines of this experiment.
//
// Links libnvinfer, libcudart, and the plugin library; no Python, no torch,
// no JIT. Two commands:
//
//   runner run   --engine E --input X.bin --shape 4,3,1088,1920 --output Y.bin
//                [--repeat N] [--meta M.json] [--stream nonblocking|legacy]
//       Deserializes E, copies X to the GPU, runs it once and writes the FP32
//       output. With --repeat, runs N more times and reports in M.json whether
//       every run reproduced the first byte for byte.
//
//   runner bench --engine-a A --engine-b B --input X.bin --shape ... --json R.json
//                [--output-a YA.bin --output-b YB.bin] [--warmup 20] [--runs 200]
//                [--trials 3] [--stream nonblocking|legacy]
//       Times both engines in one process with the protocol of
//       kronos_edge.benchmark_check.timed: WARMUP calls, then RUNS calls each
//       bracketed by CUDA events with no synchronization between them and one
//       synchronization at the end; TRIALS trials, the order of the two
//       engines reversed on odd trials; no cache flush. Writes every sample.
//       The final outputs after timing go to --output-a/--output-b so a
//       validator can check the post-timing results.
//
// The stream is a cudaStreamNonBlocking stream unless --stream legacy asks
// for the legacy default stream (what the product's Python runner uses).

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gaia_plugins.h"

using namespace nvinfer1;

namespace {

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t err__ = (call);                                                                        \
        if (err__ != cudaSuccess)                                                                          \
            throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(err__));        \
    } while (0)

class Logger : public ILogger {
public:
    void log(Severity severity, AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};

std::vector<char> readFile(std::string const& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeFile(std::string const& path, void const* data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(static_cast<char const*>(data), size);
    if (!out) throw std::runtime_error("short write to " + path);
}

std::vector<int64_t> parseShape(std::string const& text) {
    std::vector<int64_t> dims;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) dims.push_back(std::stoll(item));
    if (dims.empty()) throw std::runtime_error("empty --shape");
    return dims;
}

std::string dimsText(Dims const& d) {
    std::string s = "[";
    for (int32_t i = 0; i < d.nbDims; ++i) s += (i ? "," : "") + std::to_string(d.d[i]);
    return s + "]";
}

int64_t volume(Dims const& d) {
    int64_t v = 1;
    for (int32_t i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

// One deserialized engine with its device buffers. The runtime outlives the engine and the context.
class Engine {
public:
    Engine(std::string const& path, std::vector<int64_t> const& inputShape, Logger& logger) : mPath(path) {
        std::vector<char> blob = readFile(path);
        mRuntime.reset(createInferRuntime(logger));
        if (!mRuntime) throw std::runtime_error("createInferRuntime failed");
        mEngine.reset(mRuntime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!mEngine) throw std::runtime_error("deserializeCudaEngine failed for " + path);
        mContext.reset(mEngine->createExecutionContext());
        if (!mContext) throw std::runtime_error("createExecutionContext failed for " + path);
        int32_t nb = mEngine->getNbIOTensors();
        for (int32_t i = 0; i < nb; ++i) {
            char const* name = mEngine->getIOTensorName(i);
            if (mEngine->getTensorDataType(name) != DataType::kFLOAT)
                throw std::runtime_error(std::string("tensor ") + name + " is not FP32");
            Dims dims = mEngine->getTensorShape(name);
            if (mEngine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                if (!mInputName.empty()) throw std::runtime_error("more than one input");
                if ((size_t) dims.nbDims != inputShape.size()) throw std::runtime_error("input rank mismatch");
                for (int32_t k = 0; k < dims.nbDims; ++k)
                    if (dims.d[k] != inputShape[k])
                        throw std::runtime_error("engine input shape " + dimsText(dims) + " does not match --shape");
                mInputName = name;
                mInputBytes = volume(dims) * sizeof(float);
                CUDA_CHECK(cudaMalloc(&mInput, mInputBytes));
                if (!mContext->setTensorAddress(name, mInput)) throw std::runtime_error("setTensorAddress failed");
            } else {
                if (!mOutputName.empty()) throw std::runtime_error("more than one output");
                for (int32_t k = 0; k < dims.nbDims; ++k)
                    if (dims.d[k] < 0) throw std::runtime_error("output shape is not static");
                mOutputName = name;
                mOutputDims = dims;
                mOutputBytes = volume(dims) * sizeof(float);
                CUDA_CHECK(cudaMalloc(&mOutput, mOutputBytes));
                if (!mContext->setTensorAddress(name, mOutput)) throw std::runtime_error("setTensorAddress failed");
            }
        }
        if (mInputName.empty() || mOutputName.empty()) throw std::runtime_error("engine needs one input and one output");
    }

    ~Engine() {
        cudaFree(mInput);
        cudaFree(mOutput);
    }

    void upload(std::vector<char> const& host, cudaStream_t stream) {
        if (host.size() != mInputBytes)
            throw std::runtime_error("input file has " + std::to_string(host.size()) + " bytes, engine expects " + std::to_string(mInputBytes));
        CUDA_CHECK(cudaMemcpyAsync(mInput, host.data(), mInputBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void run(cudaStream_t stream) {
        if (!mContext->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed for " + mPath);
    }

    std::vector<char> download(cudaStream_t stream) {
        std::vector<char> host(mOutputBytes);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(host.data(), mOutput, mOutputBytes, cudaMemcpyDeviceToHost));
        return host;
    }

    Dims const& outputDims() const { return mOutputDims; }
    std::string const& path() const { return mPath; }

private:
    std::string mPath;
    std::unique_ptr<IRuntime> mRuntime;      // declared first: destroyed last
    std::unique_ptr<ICudaEngine> mEngine;
    std::unique_ptr<IExecutionContext> mContext;
    std::string mInputName, mOutputName;
    Dims mOutputDims{};
    void* mInput = nullptr;
    void* mOutput = nullptr;
    size_t mInputBytes = 0, mOutputBytes = 0;
};

struct Stream {
    cudaStream_t handle = nullptr;
    bool owned = false;
    explicit Stream(std::string const& kind) {
        if (kind == "legacy") {
            handle = nullptr;  // the legacy default stream
        } else if (kind == "nonblocking") {
            CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking));
            owned = true;
        } else {
            throw std::runtime_error("--stream must be nonblocking or legacy");
        }
    }
    ~Stream() {
        if (owned) cudaStreamDestroy(handle);
    }
};

// kronos_edge.benchmark_check.timed: warm up, then one (start, end) event pair per run, one synchronize at the end.
std::vector<float> timed(Engine& engine, cudaStream_t stream, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) engine.run(stream);
    std::vector<cudaEvent_t> starts(runs), ends(runs);
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&ends[i]));
    }
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(starts[i], stream));
        engine.run(stream);
        CUDA_CHECK(cudaEventRecord(ends[i], stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms(runs);
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], starts[i], ends[i]));
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(ends[i]);
    }
    return ms;
}

struct Stats {
    double mean, median, p95;
};

Stats stats(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    size_t n = samples.size();
    double median = n % 2 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    // numpy.percentile(q=95) default linear interpolation
    double rank = 0.95 * (n - 1);
    size_t lo = (size_t) rank;
    size_t hi = std::min(lo + 1, n - 1);
    double p95 = samples[lo] + (rank - lo) * (samples[hi] - samples[lo]);
    return {mean, median, p95};
}

std::string json(Stats const& s) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "{\"mean\": %.6f, \"median\": %.6f, \"p95\": %.6f}", s.mean, s.median, s.p95);
    return buf;
}

std::string json(std::vector<float> const& v) {
    std::string s = "[";
    char buf[32];
    for (size_t i = 0; i < v.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%.6f", v[i]);
        s += (i ? ", " : "") + std::string(buf);
    }
    return s + "]";
}

std::string quote(std::string const& s) {
    return "\"" + s + "\"";
}

struct Args {
    std::string command;
    std::vector<std::pair<std::string, std::string>> options;
    std::string get(std::string const& key, std::string const& fallback = "") const {
        for (auto const& kv : options)
            if (kv.first == key) return kv.second;
        if (fallback.empty()) throw std::runtime_error("missing " + key);
        return fallback;
    }
    bool has(std::string const& key) const {
        for (auto const& kv : options)
            if (kv.first == key) return true;
        return false;
    }
};

Args parse(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: runner run|bench --option value ...");
    Args a;
    a.command = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        if (std::strncmp(argv[i], "--", 2) != 0) throw std::runtime_error(std::string("unexpected argument ") + argv[i]);
        a.options.emplace_back(argv[i] + 2, argv[i + 1]);
    }
    if ((argc - 2) % 2 != 0) throw std::runtime_error("every option needs a value");
    return a;
}

std::string versionText() {
    return std::to_string(getInferLibVersion());
}

int cmdRun(Args const& a, Logger& logger) {
    std::vector<int64_t> shape = parseShape(a.get("shape"));
    Stream stream(a.get("stream", "nonblocking"));
    Engine engine(a.get("engine"), shape, logger);
    std::vector<char> input = readFile(a.get("input"));
    engine.upload(input, stream.handle);
    engine.run(stream.handle);
    std::vector<char> first = engine.download(stream.handle);
    writeFile(a.get("output"), first.data(), first.size());
    int repeat = std::stoi(a.get("repeat", "0"));
    int identical = 0;
    for (int i = 0; i < repeat; ++i) {
        engine.run(stream.handle);
        if (engine.download(stream.handle) == first) ++identical;
    }
    if (a.has("meta")) {
        std::ostringstream m;
        m << "{\n  \"engine\": " << quote(engine.path()) << ",\n  \"output_shape\": " << dimsText(engine.outputDims())
          << ",\n  \"repeat\": " << repeat << ",\n  \"repeats_identical\": " << identical
          << ",\n  \"deterministic\": " << ((identical == repeat) ? "true" : "false")
          << ",\n  \"stream\": " << quote(a.get("stream", "nonblocking")) << ",\n  \"tensorrt_version\": " << quote(versionText()) << "\n}\n";
        std::string text = m.str();
        writeFile(a.get("meta"), text.data(), text.size());
    }
    std::printf("ran %s: output %s, %d/%d repeats identical\n", engine.path().c_str(), dimsText(engine.outputDims()).c_str(), identical, repeat);
    return (identical == repeat) ? 0 : 2;
}

int cmdBench(Args const& a, Logger& logger) {
    std::vector<int64_t> shape = parseShape(a.get("shape"));
    int warmup = std::stoi(a.get("warmup", "20")), runs = std::stoi(a.get("runs", "200")), trials = std::stoi(a.get("trials", "3"));
    Stream stream(a.get("stream", "nonblocking"));
    Engine engineA(a.get("engine-a"), shape, logger), engineB(a.get("engine-b"), shape, logger);
    std::vector<char> input = readFile(a.get("input"));
    engineA.upload(input, stream.handle);
    engineB.upload(input, stream.handle);
    std::vector<Engine*> engines{&engineA, &engineB};
    std::vector<std::string> names{"a", "b"};
    std::vector<std::vector<float>> all(2);
    std::ostringstream trialsJson;
    trialsJson << "[";
    for (int t = 0; t < trials; ++t) {
        std::vector<int> order = (t % 2 == 0) ? std::vector<int>{0, 1} : std::vector<int>{1, 0};
        std::vector<std::vector<float>> samples(2);
        for (int idx : order) samples[idx] = timed(*engines[idx], stream.handle, warmup, runs);
        for (int idx : {0, 1}) all[idx].insert(all[idx].end(), samples[idx].begin(), samples[idx].end());
        trialsJson << (t ? ",\n" : "\n") << "    {\"order\": [" << quote(names[order[0]]) << ", " << quote(names[order[1]]) << "], "
                   << "\"stats_ms\": {\"a\": " << json(stats(samples[0])) << ", \"b\": " << json(stats(samples[1])) << "}, "
                   << "\"samples_ms\": {\"a\": " << json(samples[0]) << ", \"b\": " << json(samples[1]) << "}}";
        std::fprintf(stderr, "trial %d a %s b %s\n", t + 1, json(stats(samples[0])).c_str(), json(stats(samples[1])).c_str());
    }
    trialsJson << "\n  ]";
    Stats sa = stats(all[0]), sb = stats(all[1]);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0, runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::ostringstream out;
    out << "{\n  \"engine_a\": " << quote(engineA.path()) << ",\n  \"engine_b\": " << quote(engineB.path())
        << ",\n  \"protocol\": {\"timer\": \"C++ port of kronos_edge.benchmark_check.timed\", \"warmup\": " << warmup << ", \"runs_per_trial\": " << runs
        << ", \"trials\": " << trials << ", \"cache_flush\": false, \"stream\": " << quote(a.get("stream", "nonblocking")) << "}"
        << ",\n  \"environment\": {\"tensorrt_version\": " << quote(versionText()) << ", \"cuda_driver_version\": " << driver
        << ", \"cuda_runtime_version\": " << runtime << ", \"gpu\": " << quote(prop.name) << ", \"sm\": \"" << prop.major << "." << prop.minor << "\"}"
        << ",\n  \"aggregate_ms\": {\"a\": " << json(sa) << ", \"b\": " << json(sb) << "}"
        << ",\n  \"median_ratio_b_over_a\": " << sb.median / sa.median
        << ",\n  \"trials\": " << trialsJson.str() << "\n}\n";
    std::string text = out.str();
    writeFile(a.get("json"), text.data(), text.size());
    if (a.has("output-a")) {
        engineA.run(stream.handle);
        std::vector<char> ya = engineA.download(stream.handle);
        writeFile(a.get("output-a"), ya.data(), ya.size());
    }
    if (a.has("output-b")) {
        engineB.run(stream.handle);
        std::vector<char> yb = engineB.download(stream.handle);
        writeFile(a.get("output-b"), yb.data(), yb.size());
    }
    std::printf("a %s\nb %s\nmedian ratio b/a %.4f\n", json(sa).c_str(), json(sb).c_str(), sb.median / sa.median);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse(argc, argv);
        if (!gaia_trt_layout_register_plugins()) throw std::runtime_error("plugin registration failed");
        Logger logger;
        if (args.command == "run") return cmdRun(args, logger);
        if (args.command == "bench") return cmdBench(args, logger);
        throw std::runtime_error("unknown command " + args.command);
    } catch (std::exception const& e) {
        std::fprintf(stderr, "runner: %s\n", e.what());
        return 1;
    }
}
