import pandas as pd
threads = [8, 16, 128, 512, 1024, 2048, 4096]
Ns = [64, 128, 256, 512, 1024, 2048, 4096]

res = {
  "64" : [],
  "128" : [],
  "256" : [],
  "512" : [],
  "1024" : [],
  "2048" : [],
  "4096" : []
}

for N in Ns:
    for t_i, t in enumerate(threads):
        dir = "results/threads_in_block_" + str(t_i) + "/N" + str(N) + "/" + "result.txt"
        with open(dir) as f:
          line = f.readline()
          res[str(N)].append(double(line))
          print(res)

#df = pd.DataFrame(res)
#df.to_csv('out.csv', index=False)
