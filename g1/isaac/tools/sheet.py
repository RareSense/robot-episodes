import sys
from PIL import Image
out=sys.argv[1]; fs=sys.argv[2:]
ims=[Image.open(f).convert('RGB') for f in fs]
w,h=ims[0].size; n=len(ims); c=2 if n>1 else 1; r=(n+c-1)//c
S=Image.new('RGB',(w*c,h*r))
for i,m in enumerate(ims): S.paste(m,((i%c)*w,(i//c)*h))
S.thumbnail((1600,1600)); S.save(out)
