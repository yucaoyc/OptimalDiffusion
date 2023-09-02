from pynvml.smi import nvidia_smi
import time

nvsmi = nvidia_smi.getInstance()
begin_time = time.time()

while True:
	time.sleep(10)
	to_continue = True
	for j in range(3):
		item = nvsmi.DeviceQuery('memory.free, memory.total')['gpu'][j]['fb_memory_usage']
		if item['free']/item['total'] < 0.9: # more than 10% memory is occupied.
			to_continue = False
	
	if to_continue:
		break
	else:
		if time.time() - begin_time > 10*60: # after 10 mins, jsut a reminder
			print("Process locks the next step")
			begin_time = time.time()
