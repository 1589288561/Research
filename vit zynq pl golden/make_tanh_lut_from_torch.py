import math

FRAC = 12
SCALE = 1 << FRAC

X_MIN = -32768
X_MAX = 32767

OUT_FILE = "tanh_lut_q412_full.vh"


def q412_round(x: float) -> int:
    v = int(round(x * SCALE))
    return max(-32768, min(32767, v))


def verilog_signed_16(v: int) -> str:
    if v < 0:
        return f"-16'sd{abs(v)}"
    return f"16'sd{v}"


def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("// Auto-generated tanh LUT for Q4.12 input and output\n")
        f.write("// Input index covers the full signed 16-bit Q4.12 range\n\n")
        f.write("reg signed [15:0] tanh_lut [0:65535];\n\n")
        f.write("initial begin\n")
        for idx, xq in enumerate(range(X_MIN, X_MAX + 1)):
            yq = q412_round(math.tanh(xq / SCALE))
            f.write(f"    tanh_lut[{idx}] = {verilog_signed_16(yq)};\n")
        f.write("end\n\n")
        f.write("function automatic signed [15:0] tanh_lut_lookup;\n")
        f.write("    input signed [15:0] x;\n")
        f.write("    integer idx;\n")
        f.write("    begin\n")
        f.write("        idx = $signed(x) + 32768;\n")
        f.write("        tanh_lut_lookup = tanh_lut[idx];\n")
        f.write("    end\n")
        f.write("endfunction\n")

    print(f"Generated {OUT_FILE}")
    print("Range: full signed Q4.12")
    print("Entries: 65536")


if __name__ == "__main__":
    main()
