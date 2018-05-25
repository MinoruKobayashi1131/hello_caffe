#include <iostream>
#include <memory>
#include <chrono>
#include <stdio.h>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/timer.hpp>


using namespace caffe;
using namespace std;


int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;
    GlobalInit(&argc, &argv);


    Caffe::set_mode(Caffe::GPU);


    // Layer1 : [intput1]
    caffe::LayerParameter data_layer_param1;
    caffe::MemoryDataParameter &memory_data_param1 = (*data_layer_param1.mutable_memory_data_param());
    data_layer_param1.set_name("input1");
    data_layer_param1.set_type("MemoryData");
    data_layer_param1.add_top("data");
    data_layer_param1.add_top("dummy_label1");
    memory_data_param1.set_batch_size(1);
    memory_data_param1.set_channels(2);
    memory_data_param1.set_height(1);
    memory_data_param1.set_width(1);
    printf("CAFFE Layer1 initialized\n");


    // Layer2 : [intput2]
    caffe::LayerParameter data_layer_param2;
    caffe::MemoryDataParameter &memory_data_param2 = (*data_layer_param2.mutable_memory_data_param());
    data_layer_param2.set_name("input2");
    data_layer_param2.set_type("MemoryData");
    data_layer_param2.add_top("label");
    data_layer_param2.add_top("dummy_label2");
    memory_data_param2.set_batch_size(1);
    memory_data_param2.set_channels(2);
    memory_data_param2.set_height(1);
    memory_data_param2.set_width(1);
    printf("CAFFE Layer2 initialized\n");


    // Layer3 : [res]
    caffe::LayerParameter res_layer_param;
    caffe::ReshapeParameter &res_param = (*res_layer_param.mutable_reshape_param());
    res_layer_param.set_name("res");
    res_layer_param.set_type("Reshape");
    res_layer_param.add_bottom("data");
    res_layer_param.add_top("datax");
    caffe::BlobShape bs1;
    bs1.add_dim(1);
    bs1.add_dim(2);
    bs1.add_dim(1);
    bs1.add_dim(1);
    (*res_param.mutable_shape()) = bs1;
    printf("CAFFE Layer3 initialized\n");


    // Layer4 : [resx]
    caffe::LayerParameter resx_layer_param;
    caffe::ReshapeParameter &resx_param = (*resx_layer_param.mutable_reshape_param());
    resx_layer_param.set_name("resx");
    resx_layer_param.set_type("Reshape");
    resx_layer_param.add_bottom("label");
    resx_layer_param.add_top("labelx");
    caffe::BlobShape bs2;
    bs2.add_dim(1);
    bs2.add_dim(2);
    (*resx_param.mutable_shape()) = bs2;
    printf("CAFFE Layer4 initialized\n");


    // Layer5 : [rnn]
    caffe::LayerParameter rnn_layer_param;
    caffe::RecurrentParameter &rnn_param = (*rnn_layer_param.mutable_recurrent_param());
    caffe::FillerParameter &rnn_weight_filler_param = (*rnn_param.mutable_weight_filler());
    caffe::FillerParameter &rnn_bias_filler_param = (*rnn_param.mutable_bias_filler());
    rnn_layer_param.set_name("rnn");
    rnn_layer_param.set_type("LSTM");
    rnn_layer_param.add_bottom("data");
    rnn_layer_param.add_bottom("labelx");
    rnn_layer_param.add_top("rnn1");
    rnn_param.set_num_output(2);
    rnn_weight_filler_param.set_type("uniform");
    rnn_weight_filler_param.set_min(-0.08);
    rnn_weight_filler_param.set_max(0.08);
    rnn_bias_filler_param.set_type("constant");
    rnn_bias_filler_param.set_value(0);
    printf("CAFFE Layer5 initialized\n");


    // Layer6 : [ip1]
    caffe::LayerParameter ip_layer_param1;
    caffe::InnerProductParameter &ip_param1 = (*ip_layer_param1.mutable_inner_product_param());
    caffe::FillerParameter &ip_weight_filler1 = (*ip_param1.mutable_weight_filler());
    caffe::FillerParameter &ip_bias_filler1 = (*ip_param1.mutable_bias_filler());
    ip_layer_param1.set_name("ip1");
    ip_layer_param1.set_type("InnerProduct");
    (*ip_layer_param1.add_bottom()) = "rnn1";
    (*ip_layer_param1.add_top()) = "ip1";
    ip_param1.set_num_output(30);
    ip_weight_filler1.set_type("xavier");
    ip_bias_filler1.set_type("constant");
    printf("CAFFE Layer6 initialized\n");


    // Layer 7 : [relu1]
    caffe::LayerParameter relu_layer_param1;
    relu_layer_param1.set_name("relu1");
    relu_layer_param1.set_type("ReLU");
    (*relu_layer_param1.add_bottom()) = "ip1";
    (*relu_layer_param1.add_top()) = "ip1";
    printf("CAFFE Layer7 initialized\n");


    // Layer 8 : [ip2]
    caffe::LayerParameter ip_layer_param2;
    caffe::InnerProductParameter &ip_param2 = (*ip_layer_param2.mutable_inner_product_param());
    caffe::FillerParameter &ip_weight_filler2 = (*ip_param2.mutable_weight_filler());
    caffe::FillerParameter &ip_bias_filler2 = (*ip_param2.mutable_bias_filler());
    ip_layer_param2.set_name("ip2");
    ip_layer_param2.set_type("InnerProduct");
    (*ip_layer_param2.add_bottom()) = "ip1";
    (*ip_layer_param2.add_top()) = "ip2";
    ip_param2.set_num_output(2);
    ip_weight_filler2.set_type("xavier");
    ip_bias_filler2.set_type("constant");
    printf("CAFFE Layer8 initialized\n");


    // Layer 9 : [slicer_label]
    caffe::LayerParameter slicer_layer_param;
    caffe::SliceParameter &slicer_param = (*slicer_layer_param.mutable_slice_param());
    slicer_layer_param.set_name("slicer_label");
    slicer_layer_param.set_type("Slice");
    (*slicer_layer_param.add_bottom()) = "label";
    (*slicer_layer_param.add_top()) = "label1";
    (*slicer_layer_param.add_top()) = "label2";
    slicer_param.set_axis(1);
    slicer_param.add_slice_point(1);
    printf("CAFFE Layer9 initialized\n");


    // Layer 10 : [loss]
    caffe::LayerParameter loss_layer_param;
    caffe::NetStateRule nsr3;
    loss_layer_param.set_name("loss");
    loss_layer_param.set_type("SoftmaxWithLoss");
    (*loss_layer_param.add_bottom()) = "ip2";
    (*loss_layer_param.add_bottom()) = "label1";
    (*loss_layer_param.add_top()) = "loss";
    nsr3.set_phase(TRAIN);
    (*loss_layer_param.add_include()) = nsr3;
    printf("CAFFE Layer10 initialized\n");


    // Layer 11 : [accuracy]
    caffe::LayerParameter accuracy_layer_param;
    caffe::NetStateRule nsr4;
    accuracy_layer_param.set_name("accuracy");
    accuracy_layer_param.set_type("Accuracy");
    (*accuracy_layer_param.add_bottom()) = "ip2";
    (*accuracy_layer_param.add_bottom()) = "label1";
    (*accuracy_layer_param.add_top()) = "accuracy";
    //nsr4.set_phase(Phase::TRAIN);
    nsr4.set_phase(TRAIN);

    (*accuracy_layer_param.add_include()) = nsr4;
    printf("CAFFE Layer11 initialized\n");


    // Layer 12 : [prob]
    caffe::LayerParameter softmax_layer_param;
    caffe::NetStateRule nsr5;
    softmax_layer_param.set_name("prob");
    softmax_layer_param.set_type("Softmax");
    (*softmax_layer_param.add_bottom()) = "ip2";
    (*softmax_layer_param.add_top()) = "prob";
    nsr5.set_phase(TEST);
    (*softmax_layer_param.add_include()) = nsr5;
    printf("CAFFE Layer12 initialized\n");


    // ネットステートの作成
    caffe::NetState *state_train = new caffe::NetState();
    caffe::NetState *state_test = new caffe::NetState();


    // 学習用ネットワークの作成
    NetParameter net_param_train;
    state_train->set_phase(TRAIN);
    net_param_train.set_allocated_state(state_train);
    (*net_param_train.add_layer()) = data_layer_param1;
    (*net_param_train.add_layer()) = data_layer_param2;
    (*net_param_train.add_layer()) = res_layer_param;
    (*net_param_train.add_layer()) = resx_layer_param;
    (*net_param_train.add_layer()) = rnn_layer_param;
    (*net_param_train.add_layer()) = ip_layer_param1;
    (*net_param_train.add_layer()) = relu_layer_param1;
    (*net_param_train.add_layer()) = ip_layer_param2;
    (*net_param_train.add_layer()) = slicer_layer_param;
    (*net_param_train.add_layer()) = loss_layer_param;
    (*net_param_train.add_layer()) = accuracy_layer_param;
    (*net_param_train.add_layer()) = softmax_layer_param;
    printf("CAFFE TRAIN net initialized\n");


    // 予測用ネットワークの作成
    NetParameter net_param_test;
    state_test->set_phase(TEST);
    net_param_test.set_allocated_state(state_test);
    (*net_param_test.add_layer()) = data_layer_param1;
    (*net_param_test.add_layer()) = data_layer_param2;
    (*net_param_test.add_layer()) = res_layer_param;
    (*net_param_test.add_layer()) = resx_layer_param;
    (*net_param_test.add_layer()) = rnn_layer_param;
    (*net_param_test.add_layer()) = ip_layer_param1;
    (*net_param_test.add_layer()) = relu_layer_param1;
    (*net_param_test.add_layer()) = ip_layer_param2;
    (*net_param_test.add_layer()) = slicer_layer_param;
    (*net_param_test.add_layer()) = loss_layer_param;
    (*net_param_test.add_layer()) = accuracy_layer_param;
    (*net_param_test.add_layer()) = softmax_layer_param;
    printf("CAFFE TEST net initialized\n");


    // ソルバーを設定
    SolverParameter solver_param;
    (*solver_param.mutable_type()) = "SGD";
    solver_param.set_lr_policy("fixed");
    solver_param.set_base_lr(0.01f);
    solver_param.set_gamma(0.1);
    solver_param.set_display(0);
    (*solver_param.mutable_net_param()) = net_param_train;
    //caffe::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::shared_ptr<Solver<float>> solver(SolverRegistry<float>::CreateSolver(solver_param));

    // 学習用ネットワーク
    caffe::shared_ptr<caffe::Net<float>> net_train;
    net_train = solver->net();


    // 入力データ領域の準備
    const auto input_layer1 =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net_train->layer_by_name("input1"));
    const auto input_layer2 =
            boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net_train->layer_by_name("input2"));


    // 入力データ(EX-OR)
    const int data_size = 4;
    const int input_num = 2;
    float input_data[data_size][input_num] = {{0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f}};
    float label[data_size][input_num] = { {0.0f,0.0f}, {1.0f,1.0f}, {1.0f,1.0f}, {0.0f,0.0f}};
    float dummy_label[data_size] = {0.0f,0.0f,0.0f,0.0f};


    input_layer1->Reset((float*)input_data,(float*)dummy_label,data_size);
    input_layer2->Reset((float*)label,(float*)dummy_label,data_size);


    // Initialize bias for the forget gate to 5 as described in the clockwork RNN paper
    const caffe::vector<caffe::shared_ptr<Layer<float> > >& layers = net_train->layers();
    for (int i = 0; i < layers.size(); ++i) {
        if (strcmp(layers[i]->type(), "LSTM") != 0) {
            continue;
        }
        const int h = layers[i]->layer_param().recurrent_param().num_output();
        caffe::shared_ptr<Blob<float> > bias = layers[i]->blobs()[2];
        caffe_set(h, 5.0f, bias->mutable_cpu_data() + h);
    }


    // 学習
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();

    float loss = 10000.0f;
    float accuracy = 0.0f;
    int iter = 0;
    while((loss > 0.0005) || (accuracy < 0.95f))
    {
        // 学習の実行
        solver->Step(1);

        loss = net_train->blob_by_name("loss")->cpu_data()[0];
        accuracy = net_train->blob_by_name("accuracy")->cpu_data()[0];
        if(iter < 0.0005)
        {
            printf("iter : %d loss : %f  accuracy : %f\n",iter,loss,accuracy);
        }
        else
        {
            if ((iter % 1000) == 0)
                printf("iter : %d loss : %f  accuracy : %f\n",iter,loss,accuracy);
        }
        iter++;
    }
    net_train->blob_by_name("loss")->cpu_data();
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    printf("learning time : %f[msec]\n",elapsed);


    // 学習結果を書き込み
    char chr[256];
    sprintf(chr,"_iter_%d.caffemodel",iter);
    string str = chr;
    solver->Snapshot();


    // 学習結果を読み込み
    caffe::shared_ptr<Net<float>> net_test(new Net<float>(net_param_test));
    net_test->CopyTrainedLayersFrom(str);


    // 予測用入力データ領域の準備
    const auto input_test_layer =
            boost::dynamic_pointer_cast<MemoryDataLayer<float>>(net_test->layer_by_name("input1"));
    const auto dummy_test_layer =
            boost::dynamic_pointer_cast<MemoryDataLayer<float>>(net_test->layer_by_name("input2"));


    // 予測結果の表示
    for (int idx = 0; idx < data_size; idx++)
    {
        float loss;
        input_test_layer->Reset((float*)input_data[idx], (float*)dummy_label, 1);
        dummy_test_layer->Reset((float*)dummy_label, (float*)dummy_label, 1);

        // 予測の実行
        const vector<Blob<float>*> result = net_test->Forward(&loss);

        const auto data = result[result.size() - 1]->cpu_data();
        int ans = (data[0] > data[1]) ? 0:1;
        std::cout << idx << " : " << data[0] << ", " << data[1] << " (ans) : " << ans << std::endl;
    }
}
