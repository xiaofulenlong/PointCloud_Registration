import subprocess

cmd = []
cmd.append("python ./metrics.py -p1 ./data/bunny-pcd/bunny-pcd-1.ply  -p2 ./result/bunny/bunny-2-registered.ply")
cmd.append("python ./metrics.py -p1 ./data/bunny-pcd/bunny-pcd-1.ply  -p2 ./result/bunny/bunny-3-registered.ply")

cmd.append("python ./metrics.py  -p1 ./data/room-pcd/room-pcd-1.ply  -p2 ./result/room/room-2-registered.ply")
cmd.append("python ./metrics.py  -p1 ./data/room-pcd/room-pcd-1.ply  -p2 ./result/room/room-3-registered.ply")

cmd.append("python ./metrics.py  -p1 ./data/temple-pcd/temple-pcd-1.ply  -p2 ./result/temple/temple-2-registered.ply")
cmd.append("python ./metrics.py  -p1 ./data/temple-pcd/temple-pcd-1.ply  -p2 ./result/temple/temple-2-registered.ply")

for i in range(len(cmd)): 
    subprocess.run(cmd[i])



