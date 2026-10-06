"""
Probe P1 Part 2: Inspect TFLite Subgraph Operators in Section 10 (tf_lite_prefill_decode)
Determines:
1. Exact input/output tensor shapes and types for 'prefill_1024' and 'decode' signatures.
2. Operator sequence: are layers 15-34 evaluated for sequence_length=1024 or sliced?
"""

import sys
import os

MODEL_PATH = r"C:\Users\alefita\.litert-lm\cache\huggingface\litert-community\gemma-4-E2B-it-litert-lm\gemma-4-E2B-it.litertlm"

def main():
    print("[+] Extracting TFLite buffer for Section 10...")
    with open(MODEL_PATH, "rb") as f:
        f.seek(1725530112)
        # Read the TFLite model bytes
        tflite_bytes = f.read(2543805296 - 1725530112)
    
    print(f"[+] Loaded TFLite model bytes: {len(tflite_bytes)} bytes ({len(tflite_bytes)/(1024*1024):.2f} MB)")
    
    # Let's inspect using tflite python library if available or flatbuffers
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(tflite_bytes, 0)
        print(f"[+] TFLite Model parsed successfully! Subgraphs count: {model.SubgraphsLength()}")
        
        for s_idx in range(model.SubgraphsLength()):
            sg = model.Subgraphs(s_idx)
            name = sg.Name().decode('utf-8') if sg.Name() else f"subgraph_{s_idx}"
            print(f"\n--- Subgraph {s_idx}: '{name}' ---")
            print(f"    Tensors: {sg.TensorsLength()}, Operators: {sg.OperatorsLength()}")
            print(f"    Inputs:  {sg.InputsLength()}, Outputs: {sg.OutputsLength()}")
            
            # Print inputs
            for i in range(min(sg.InputsLength(), 10)):
                t_idx = sg.Inputs(i)
                t = sg.Tensors(t_idx)
                shape = [t.Shape(d) for d in range(t.ShapeLength())]
                t_name = t.Name().decode('utf-8') if t.Name() else ""
                print(f"      Input [{i}]: tensor #{t_idx} '{t_name}' shape={shape}")
            if sg.InputsLength() > 10:
                print(f"      ... and {sg.InputsLength()-10} more inputs")
                
            # Print outputs
            for i in range(min(sg.OutputsLength(), 10)):
                t_idx = sg.Outputs(i)
                t = sg.Tensors(t_idx)
                shape = [t.Shape(d) for d in range(t.ShapeLength())]
                t_name = t.Name().decode('utf-8') if t.Name() else ""
                print(f"      Output [{i}]: tensor #{t_idx} '{t_name}' shape={shape}")
            if sg.OutputsLength() > 10:
                print(f"      ... and {sg.OutputsLength()-10} more outputs")
                
            # Scan operators in subgraph to see slice / sequence dimensions
            # Count ops
            op_codes = {}
            for op_idx in range(sg.OperatorsLength()):
                op = sg.Operators(op_idx)
                code_idx = op.OpcodeIndex()
                opcode = model.OperatorCodes(code_idx)
                from tflite.BuiltinOperator import BuiltinOperator
                op_name = None
                for k, v in BuiltinOperator.__dict__.items():
                    if v == opcode.BuiltinCode():
                        op_name = k
                        break
                op_name = op_name or f"OP_{opcode.BuiltinCode()}"
                op_codes[op_name] = op_codes.get(op_name, 0) + 1
            print(f"    Operator breakdown: {dict(sorted(op_codes.items(), key=lambda x: -x[1])[:8])}")

    except ImportError:
        print("[-] 'tflite' package not directly installed, inspecting via flatbuffers schema or raw tags...")
        # Check if tensorflow or tflite exists in site-packages
        import subprocess
        # Search for signature defs in buffer
        import re
        sigs = re.findall(rb'prefill_\d+|decode', tflite_bytes[:1000000])
        print(f"Signatures matched: {set(sigs)}")

if __name__ == "__main__":
    main()
