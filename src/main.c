#include <fann.h>

void fann_try_out_bank(void)
{
  struct fann *ann;
  struct fann_train_data *train_data, *test_data;
  unsigned int num_layers = 4;
  enum fann_train_enum training_algorithm = FANN_TRAIN_BATCH;
  unsigned int bit_fail_train, bit_fail_test;
  float mse_train, mse_test;
  fann_type input[32] = {0.031033, 0.033563, 0.030325, 0.029705, 0.014498, 0.056337,
                         0.011697, 0.035937, 0.029669, 0.028296, 0.014981, 0.026507,
                         0.015139, 0.015841, 0.035542, 0.030916, 0.017335, 0.033760,
                         0.027308, 0.020122, 0.023476, 0.110401, 0.038743, 0.014973,
                         0.005865, 0.086911, 0.038549, 0.001030, 0.019641, 0.133891,
                         0.038088, 0.157380};
  fann_type *output;

  printf("Reading data.\n");

  train_data = fann_read_train_from_file("../datasets/bank32fm.train");
  test_data  = fann_read_train_from_file("../datasets/bank32fm.test");


  if(!train_data)
  {
      printf("train data failed\n");
      return;
  }

  if(!test_data)
  {
      printf("test data failed\n");
      return;
  }

  printf("train data\n");
  fann_scale_train_data(train_data, -1, 1);
  printf("train data 2:\n");
  fann_scale_train_data(test_data, -1, 1);

  ann = fann_create_standard(num_layers, fann_num_input_train_data(train_data), 8, 9, fann_num_output_train_data(train_data));
  fann_set_training_algorithm(ann, training_algorithm);
  fann_set_activation_function_layer(ann, FANN_ELLIOT_SYMMETRIC, 0);
  fann_set_activation_function_layer(ann, FANN_ELLIOT_SYMMETRIC, 1);
  fann_set_activation_function_layer(ann, FANN_ELLIOT_SYMMETRIC, 2);
  fann_set_activation_function_output(ann, FANN_LINEAR);
  fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);

  fann_set_bit_fail_limit(ann, (fann_type)0.15);
  fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);

  printf("parameters:\n");
  fann_print_parameters(ann);

  fann_train_on_data(ann, train_data, 3000, 50, 0.05);

  printf("connections:\n");
  fann_print_connections(ann);

  mse_train      = fann_test_data(ann, train_data);
  bit_fail_train = fann_get_bit_fail(ann);
  mse_test       = fann_test_data(ann, test_data);
  bit_fail_test  = fann_get_bit_fail(ann);

  printf("\nTrain error: %f, Train bit-fail: %d, Test error: %f, Test bit-fail: %d\n\n",
     mse_train, bit_fail_train, mse_test, bit_fail_test);

  output = fann_run(ann, input);
  printf("output %f\n", *output);

  fann_destroy_train(train_data);
  fann_destroy_train(test_data);
  fann_destroy(ann);
}

int main()
{
    fann_try_out_bank();
    return 0;
}
