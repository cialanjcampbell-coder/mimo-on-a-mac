import sys
n=int(sys.argv[1]) if len(sys.argv)>1 else 700
body="".join(f"def func_{i}(x):\n    return x*{i}+{i%7}\n\n" for i in range(n))
print(body+"\nWhich function returns x*123+4? Answer with the name only.")
