#Inference with pytorch
## FP32
```
python scripts\infer.py --checkpoint models\PackNet01_MR_selfsup_D.ckpt --input media\tests\ddad.png --output output\pytorch-16.png
``` 

## FP16
```
python scripts\infer.py --checkpoint models\PackNet01_MR_selfsup_D.ckpt --input media\tests\ddad.png --output output\pytorch-16.png --half
``` 

# Convert to ONNX
```
pyton onnx_packnet\convert_to_onnx.py --output output\packnet.onnx --checkpoint models\PackNet01_MR_selfsup_D.ckpt
```

# Convert to TRT
FP32 
```
D:\lib\TensorRT-8.2.3.0\bin\trtexec.exe --workspace=2000 --onnx=output/packnet.onnx --saveEngine=output/packnet-32.trt
```

FP16 
```
D:\lib\TensorRT-8.2.3.0\bin\trtexec.exe --workspace=2000 --onnx=output/packnet.onnx --saveEngine=output/packnet-16.trt --fp16
```

Best precision
```
D:\lib\TensorRT-8.2.3.0\bin\trtexec.exe --workspace=2000 --onnx=output/packnet.onnx --saveEngine=output/packnet-best.trt --best
```


#Inference with TRT
```
python onnx_packnet\infer.py --input media\tests\ddad.png --engine output\ENGINE_FILE
```
```ENGINE_FILE``` - one of the TRT engine files created in the steps above. 