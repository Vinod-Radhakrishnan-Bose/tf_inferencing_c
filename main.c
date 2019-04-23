#include <stdio.h>
#include <stdlib.h>
#include "inference_engine.h"

int main() {

    // Hardcode the model filename for now. 
    // Assumptions: This model takes in a 0-dim float tensor of size 1 and returns a 0-dim float tensor of size 1.
    static char *graph_file_name = "inference.pb";
    inference_engine_t t_inference_engine; // structure holding all information needed for performing information.
    float input_data[] = {3.0f}; // Feed dummy data. This will come from external application in the future.
    float output_results;

    // Initialize the inference engine by providing the model graph file as input
    InitializeInferenceEngine(&t_inference_engine, 
                              graph_file_name,  // model file name
                              "my_placeholder", // input node name
                              "my_dense3/kernel"); // output node name

    // Run this within a loop for every fresh data sample. Fresh input data should be provided
    // within input_data, and outputs can be fetched from output_results.
    {
        t_inference_engine.LoadData(&t_inference_engine, 
                                    input_data);
        t_inference_engine.RunInference(&t_inference_engine);
        t_inference_engine.RetrieveResult(&t_inference_engine, &output_results);
        printf("output = %f\n", output_results);
    }
    
    // Terminate the inference engine when the application ends
    t_inference_engine.TerminateInferenceEngine(&t_inference_engine);
    
    //Exit
    return 0;
}

