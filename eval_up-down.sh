echo "On Robust-COCO"
python  eval.py   --model   ./log/Robust-COCO/cexe-sup-kl-w0/model-best.pth   --infos_path  ./log/Robust-COCO/cexe-sup-kl-w0/infos_cexe-sup-kl-w0-best.pkl  --beam_size  2  --id  cexe-sup-kl-w0  --dataset  robust-coco 
python  eval.py   --model   ./log/Robust-COCO/cexe-sup-kl-w0.5/model-best.pth   --infos_path  ./log/Robust-COCO/cexe-sup-kl-w0.5/infos_cexe-sup-kl-w0.5-best.pkl  --beam_size  2  --id  cexe-sup-kl-w0.5  --dataset  robust-coco 
python  eval.py   --model   ./log/Robust-COCO/lsxe-0.5/model-best.pth   --infos_path  ./log/Robust-COCO/lsxe-0.5/infos_lsxe-0.5-best.pkl  --beam_size  2  --id  lsxe-0.5  --dataset  robust-coco 
echo "On COCO"
python  eval.py   --model   ./log/COCO/cexe-sup-kl-w0-coco/model-best.pth   --infos_path  ./log/COCO/cexe-sup-kl-w0-coco/infos_cexe-sup-kl-w0-coco-best.pkl   --beam_size  3  --id  cexe-sup-kl-w0-coco  --dataset  coco 
python  eval.py   --model   ./log/COCO/cexe-sup-kl-w0.2-coco/model-best.pth   --infos_path  ./log/COCO/cexe-sup-kl-w0.2-coco/infos_cexe-sup-kl-w0.2-coco-best.pkl   --beam_size  3  --id  cexe-sup-kl-w0.2-coco  --dataset  coco 


