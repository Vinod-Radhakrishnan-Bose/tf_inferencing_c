#include <c_api.h> // Make sure the path to this file is correct.

typedef struct t_inference_engine{
	TF_Status *status;
	
	TF_Graph *graph;
	TF_Buffer graph_def;
	TF_ImportGraphDefOptions *graph_opts;
	
	TF_SessionOptions *opts;
	TF_Session *session;
	
	TF_Operation* input_op;
	TF_Output input_tensor;
	TF_Tensor *input_value;
	
	TF_Operation* output_op;
	TF_Output output_tensor;
	TF_Tensor *output_value;
	
	void (*LoadData)(struct t_inference_engine *, float *);
	void (*RunInference)(struct t_inference_engine *);
	void (*RetrieveResult)(struct t_inference_engine *, float *);
	void (*TerminateInferenceEngine)(struct t_inference_engine *);
} inference_engine_t;

extern void InitializeInferenceEngine(inference_engine_t *pt_inference_engine, 
                                      char *graph_file_name,
									  char *input_node_name,
							          char *output_node_name);
