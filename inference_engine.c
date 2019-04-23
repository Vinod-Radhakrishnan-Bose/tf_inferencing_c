#include <stdio.h>
#include <c_api.h> // Make sure the path to this file is correct.
#include <stdlib.h>
#include "inference_engine.h"

static TF_Buffer load_graph_file(const char *filename);
static void tf_free_tensor(void *data, size_t len, void* arg);
static void LoadData(inference_engine_t *pt_inference_engine, float *input);
static void RunInference(inference_engine_t *pt_inference_engine);
static void RetrieveResult(inference_engine_t *pt_inference_engine, float *pt_output_results);
static void TerminateInferenceEngine(inference_engine_t *pt_inference_engine);

// Utility for loading a TF model def file
TF_Buffer load_graph_file(const char *filename)
{
    FILE *fp;
    TF_Buffer buff;
    if ((fp = fopen(filename, "r")) == NULL) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    long s = ftell(fp);
    char *data = malloc(s);

    rewind(fp);
    fread(data, 1, s, fp);
    fclose(fp);

    buff.data = data;
    buff.length = s;
    return buff;
}

void tf_free_tensor(void *data, size_t len, void* arg)
{
    free(data);
}

void InitializeInferenceEngine(inference_engine_t *pt_inference_engine, 
                               char *graph_file_name,
							   char *input_node_name,
							   char *output_node_name)
{
	// Variable to hold status for all TF API calls.
    pt_inference_engine->status = TF_NewStatus();
	
	//Load graph
    pt_inference_engine->graph = TF_NewGraph();
    pt_inference_engine->graph_def = load_graph_file(graph_file_name);
    pt_inference_engine->graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(pt_inference_engine->graph, &pt_inference_engine->graph_def, pt_inference_engine->graph_opts, pt_inference_engine->status);
    if (TF_GetCode(pt_inference_engine->status) != TF_OK) {
        fprintf(stderr, "Error: %s\n", TF_Message(pt_inference_engine->status));
        exit(EXIT_FAILURE);
    }
	
	// Create a session
    pt_inference_engine->opts = TF_NewSessionOptions();
    pt_inference_engine->session     = TF_NewSession(pt_inference_engine->graph, pt_inference_engine->opts, pt_inference_engine->status);
    if (TF_GetCode(pt_inference_engine->status) != TF_OK) {
        fprintf(stderr, "Error: %s\n", TF_Message(pt_inference_engine->status));
        exit(EXIT_FAILURE);
    }
	
	//Set up input tensor and input data
	pt_inference_engine->input_op = TF_GraphOperationByName(pt_inference_engine->graph, input_node_name);
	pt_inference_engine->input_value = TF_AllocateTensor(TF_FLOAT, NULL, 0, sizeof(float));
	pt_inference_engine->input_tensor.oper = pt_inference_engine->input_op;
	pt_inference_engine->input_tensor.index = 0;

	//Set up output tensor and pointer to output values
	pt_inference_engine->output_op = TF_GraphOperationByName(pt_inference_engine->graph, output_node_name);
	pt_inference_engine->output_tensor.oper = pt_inference_engine->output_op;
	pt_inference_engine->output_tensor.index = 0;

	//Set up function pointers
	pt_inference_engine->LoadData = &LoadData;
	pt_inference_engine->RunInference = &RunInference;
	pt_inference_engine->RetrieveResult = &RetrieveResult;
	pt_inference_engine->TerminateInferenceEngine = &TerminateInferenceEngine;
}

static void LoadData(inference_engine_t *pt_inference_engine, float *input)
{
	((float *)TF_TensorData(pt_inference_engine->input_value))[0] = input[0]; 
}

static void RunInference(inference_engine_t *pt_inference_engine)
{
	//Run the session
    TF_SessionRun(pt_inference_engine->session, /* Session */
                  NULL, /* RunOptions */ 
                  &pt_inference_engine->input_tensor, &pt_inference_engine->input_value, 1,   /* Input tensors, data and length */ 
                  &pt_inference_engine->output_tensor, &pt_inference_engine->output_value, 1, /* Output tensors, pointer to results and length */ 
                  NULL, 0, /* Target operations */ 
                  NULL, /* RunMetadata */ 
                  pt_inference_engine->status /* Output status */ 
				  );
}

static void RetrieveResult(inference_engine_t *pt_inference_engine, float *pt_output_results)
{
	if (NULL != pt_inference_engine->output_value)
	{
		pt_output_results[0] = *((float *)(TF_TensorData(pt_inference_engine->output_value)));
	}
}

static void TerminateInferenceEngine(inference_engine_t *pt_inference_engine)
{
	//Clean up
    TF_DeleteImportGraphDefOptions(pt_inference_engine->graph_opts);
    TF_DeleteSession(pt_inference_engine->session, pt_inference_engine->status);
    TF_DeleteStatus(pt_inference_engine->status);
    TF_DeleteSessionOptions(pt_inference_engine->opts);
    TF_DeleteGraph(pt_inference_engine->graph);
}
