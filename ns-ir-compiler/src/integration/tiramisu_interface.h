#ifndef TIRAMISU_NSIR_INTERFACE_H
#define TIRAMISU_NSIR_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// Pointer to holding the opaque libtorch or PyObject model state
typedef void* NSIRModelHandle;

/**
 * Loads the exported PyTorch model (TorchScript or RPC stub client)
 * Returns the opaque handler context.
 */
NSIRModelHandle load_nsir_model(const char* model_path);

/**
 * Executes a sequence prediction against the NS-IR predictive heuristic graph.
 * `ir_json` - Serialized structure of the program loop nest
 * `transform_json` - List array of requested compiler passes
 */
float predict_speedup(NSIRModelHandle handle, const char* ir_json, const char* transform_json);

/**
 * Top level scheduling replacement. Uses Monte-Carlo/Beam Search targeting 
 * `predict_speedup`. Allocates and returns char array with best schedule sequence string.
 */
char* auto_schedule(NSIRModelHandle handle, const char* program_ir, int search_budget);

#ifdef __cplusplus
}
#endif

#endif // TIRAMISU_NSIR_INTERFACE_H
