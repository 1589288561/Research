`ifndef FIXED_PARAMS_VH
`define FIXED_PARAMS_VH

`define WIDTH 16
`define FRAC  12

`define T 50
`define D 64
`define HEADS 4
`define HEAD_DIM 16
`define DFF 128
`define N_CLASSES 10

`define MAT_T_D_W   (`T*`D*`WIDTH)
`define MAT_T_DFF_W (`T*`DFF*`WIDTH)
`define VEC_D_W     (`D*`WIDTH)
`define VEC_DFF_W   (`DFF*`WIDTH)
`define VEC_C_W     (`N_CLASSES*`WIDTH)
`define W_D_D_W     (`D*`D*`WIDTH)
`define W_DFF_D_W   (`DFF*`D*`WIDTH)
`define W_D_DFF_W   (`D*`DFF*`WIDTH)
`define W_C_D_W     (`N_CLASSES*`D*`WIDTH)

// Legacy aliases keep the existing module ports/testbenches readable while
// the dimensions now come from the full-scale MNIST ViT block.
`define MAT5X4_W `MAT_T_D_W
`define MAT5X8_W `MAT_T_DFF_W
`define VEC4_W   `VEC_D_W
`define VEC8_W   `VEC_DFF_W
`define W4X4_W   `W_D_D_W
`define W8X4_W   `W_DFF_D_W
`define W4X8_W   `W_D_DFF_W

`define ONE   (1 <<< `FRAC)
`define EPS   16'sd1

// Select element idx from a packed bus. idx=0 maps to the least-significant slice.
`define ELEM(bus, idx) bus[((idx)+1)*`WIDTH-1 -: `WIDTH]

`endif
