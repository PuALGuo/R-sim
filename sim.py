## a simple sim for rram
import numpy as np
class CrossBar:
    def __init__(self):
        self.lock = [False,False,False]  ## for load compute store
        self.inp_buffer = [np.zeros((1,128))] * (16*16)
        # self.out_buffer = [np.zeros((1,128))] * (14*14)
        self.acc_buffer = [np.zeros((1,128))] * (14*14)
        self.wgt_buffer = [np.zeros((128,128))] * (3*3)
        self.rram_buffer = [np.zeros((128,128))] * (3*3)
        self.buffer = {} ##addr:(size)
        ##
    def RRAMBufferAlloc(self,size,addr):
        self.buffer[addr] = size
        return 1
    def RRAMBufferCopy(self, from_, from_offset, to, to_offset, size, kind_mask):
        ## 
        # print(self.buffer[to])
        # print(size)
        assert(self.buffer[to] == size)
        return 1
    def RRAMBufferFree(self, addr):
        del self.buffer[addr]
        return 1
    def RRAMDepPush(self,l1,l2):
        if self.lock[l1]:
            print("%d is used, please wait for a while"%(l1))
            return 0
        if self.lock[l2]:
            print("%d is used, please wait for a while"%(l2))
            return 0
        self.lock[l1],self.lock[l2] = True, True
        return 1
    def RRAMDepPop(self,l1,l2):
        if not self.lock[l1]:
            print("want to release a lock %d which is not used" %(l1))
            return 0
        if not self.lock[l2]:
            print("want to release a lock %d which is not used" %(l2))
            return 0
        self.lock[l1],self.lock[l2] = False, False
        return 1
    def rram_MAC(self,out_index,inp_index,wgt_index,rst_n):
        if rst_n == 0:
            self.acc_buffer[out_index] = np.zeros((1,128))
        else:
            # print(inp_index)
            # print(wgt_index)
            self.acc_buffer[out_index] += np.matmul(self.inp_buffer[inp_index],self.rram_buffer[wgt_index])
        return 1
    def RRAMLoadBuffer2D(self,src,src_addr,src_elem_offset,x_size,y_size,x_stride,x_pad_before,y_pad_before, x_pad_after, y_pad_after, dst_sram_index, dst_memory_type):
        assert (src_addr in self.buffer.keys())
        if dst_memory_type == 2: ## inp
            self.inp_buffer = [np.zeros((1,128))] * ((x_size + x_pad_after + x_pad_before) * (y_size + y_pad_after + y_pad_before) )
            for i in range(y_size):
                for j in range(x_size):
                    self.inp_buffer[dst_sram_index+(i+y_pad_before)*(x_stride+x_pad_before+x_pad_after)+(j+x_pad_before)] = src[src_elem_offset + i*x_size + j]
            return 1
        elif dst_memory_type == 1:
            assert(x_pad_before == 0)
            assert(x_pad_after  == 0)
            assert(y_pad_before == 0)
            assert(y_pad_after  == 0)
            self.wgt_buffer = [0] * (x_size * y_size)
            for i in range(y_size):
                for j in range(x_size):
                    self.wgt_buffer[i*x_stride+j] = src[src_elem_offset + i*x_stride + j] 
            return 1
        else:
            print("invalid load")
            return 0
    def RRAMStoreBuffer2D(self,dst,dst_addr,src_sram_index,src_memory_type,dst_elem_offset,x_size,y_size,x_stride):
        assert(src_memory_type == 4)
        assert(dst_addr in self.buffer.keys())
        for i in range(y_size):
            for j in range(x_size):
                dst[dst_elem_offset + i*x_stride + j] = self.acc_buffer[src_sram_index + i*x_stride + j]
        return 1
    def SetTile(self,n,h,w,kw,kh,ic,oc):
        self.batch = n
        self.height = h
        self.width = w
        assert(kw == kh)
        self.kernel = kw
        self.in_channel = ic
        self.out_channel = oc
        return 1
    def LWGT(self,weight_index,rram_index,rst_n):
        self.rram_buffer[rram_index] = 0 if rst_n == 0 else self.wgt_buffer[weight_index]
        return 1
    
if __name__ == '__main__':
    f = open('./code/202008231455.txt','r')
    lines = f.readlines()
    inp = np.random.randint(-128,127,(1,256,14,14))
    wgt = np.random.randint(-128,127,(256,256,3,3))
    from topi.testing import conv2d_nchw_python
    target = conv2d_nchw_python(inp,wgt,1,1)
    target = target.reshape(128,14*14*2).transpose(1,0)
    inp = inp.reshape((128,14*14*2)).transpose(1,0)
    wgt = wgt.reshape((128,2,128,2,3,3)).transpose(0,2,1,3,4,5).reshape((128,128,2*2*3*3)).transpose(2,1,0)
    out = np.zeros((2*14*14,128))
    device = CrossBar()
    for line in lines:
        if "RRAMBufferAlloc" in line:
            tmp = line.split()
            device.RRAMBufferAlloc(int(tmp[1]),tmp[2])
        elif "RRAMBufferCopy" in line:
            tmp = line.split()
            device.RRAMBufferCopy(tmp[1],int(tmp[2]),tmp[3],int(tmp[4]),int(tmp[5]),int(tmp[6]))
        elif "RRAMBufferFree" in line:
            tmp = line.split()
            device.RRAMBufferFree(tmp[1])
        elif "DepPush" in line:
            lock = line[line.index('(')+1:line.index(')')].replace(',',' ').split()
            assert(len(lock) == 2)
            l1, l2 = lock
            assert(device.RRAMDepPush(int(l1)-1,int(l2)-1))
        elif "DepPop" in line:
            lock = line[line.index('(')+1:line.index(')')].replace(',',' ').split()
            assert(len(lock) == 2)
            l1, l2 = lock
            assert(device.RRAMDepPop(int(l1)-1,int(l2)-1))
        elif "rram_MAC" in line:
            tmp = line[9:].split()
            out_index,inp_index,wgt_index,rst_n = [int(x) for x in tmp]
            assert(device.rram_MAC(out_index,inp_index,wgt_index,rst_n))
        elif "LoadBuffer2D" in line:
            addr = line[line.index('[') + 1:line.index(']')]
            tmp = line[line.index(' ')+1:][16:].split()
            src_elem_offset,x_size,y_size,x_stride,x_pad_before,y_pad_before, x_pad_after, y_pad_after, dst_sram_index, dst_memory_type = [int(x) for x in tmp]
            if dst_memory_type == 2:
                assert(device.RRAMLoadBuffer2D(inp,addr,src_elem_offset,x_size,y_size,x_stride,x_pad_before,y_pad_before, x_pad_after, y_pad_after, dst_sram_index, dst_memory_type))
            elif dst_memory_type == 1:
                assert(device.RRAMLoadBuffer2D(wgt,addr,src_elem_offset,x_size,y_size,x_stride,x_pad_before,y_pad_before, x_pad_after, y_pad_after, dst_sram_index, dst_memory_type))
            else:
                assert(0)
        elif "Set" in line:
            tmp = line[9:].split()
            n,h,w,kw,kh,ic,oc = [int(x) for x in tmp]
            assert(device.SetTile(n,h,w,kw,kh,ic,oc))
        elif "LWGT" in line:
            tmp = line[5:].split()
            weight_index,rram_index,rst_n = [int(x) for x in tmp]
            assert(device.LWGT(weight_index,rram_index,rst_n))
        elif "StoreBuffer2D" in line:
            addr = line[line.index('[')+1:line.index(']')]
            tmp = line[line.index(']')+1:][17:].split()
            src_sram_index,src_memory_type,dst_elem_offset,x_size,y_size,x_stride = [int(x) for x in tmp]
            assert(device.RRAMStoreBuffer2D(out,addr,src_sram_index,src_memory_type,dst_elem_offset,x_size,y_size,x_stride))
        elif "RRAMUopLoopBegin" in line or "RRAMUopLoopEnd" in line:
            pass
        elif "RRAMUopPush" in line or "RRAMPushALUOp" in line:
            pass
        elif "RRAMSynchronize" in line:
            print("start sim")
        else:
            print(line)
            print("miss command")
            exit()
    import tvm
    tvm.testing.assert_allclose(target, out)
    print('everthing is ok')
    print(out[12]) 

    