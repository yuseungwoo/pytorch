#include <torch/torch.h>
#include <torch/script.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

// Where to find the MNIST dataset.
const char* kDataRoot = "./data/MNIST/raw";

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

template <typename DataLoader>
void test(
    torch::jit::script::Module &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    std::vector<torch::jit::IValue> input{data};
    auto output = model.forward(input).toTensor();
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}

int main(int argc, const char* argv[]){
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Inferencing on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Inferencing on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  
  torch::jit::script::Module model;
    try{
        model=torch::jit::load(argv[1]);
        model.to(device);
        std::cout << "loading completed\n";
    }
    catch(const c10::Error &e){
        std::cerr << "error loading the model\n";
        return -1;
    }

  std::cout << "model load Done\n";

  auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  test(model, device, *test_loader, test_dataset_size);

  return 0;
}
