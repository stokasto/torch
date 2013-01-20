#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Relu.c"
#else

static int nn_(Relu_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  
  TH_TENSOR_APPLY2(real, output, real, input,				\
                   if (!neg && (*input_data) > lambda) *output_data = *input_data - lambda; \
                   else if (neg && (*input_data) < -lambda) *output_data = *input_data + lambda; \
                   else *output_data = 0;);
  return 1;
}

static int nn_(Relu_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real lambda = luaT_getfieldchecknumber(L, 1, "lambda");
  int neg = luaT_getfieldcheckboolean(L, 1, "neg");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,	\
                   if ((!neg && (*input_data) > lambda) || (neg && (*input_data) < -lambda)) \
		     *gradInput_data = (*gradOutput_data);		\
		   else							\
		     *gradInput_data = 0;				\
    );
  return 1;
}

static const struct luaL_Reg nn_(Relu__) [] = {
  {"Relu_updateOutput", nn_(Relu_updateOutput)},
  {"Relu_updateGradInput", nn_(Relu_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Relu_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Relu__), "nn");
  lua_pop(L,1);
}

#endif
