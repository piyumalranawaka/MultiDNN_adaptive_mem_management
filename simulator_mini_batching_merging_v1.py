################################################
#Systolic Array Simulator
#Extended from SCALESIM Simulator
#Authour: Piyumal Ranawaka
################################################

import configparser as cp
import os
from absl import flags
import time
import pandas as pd
from queue import Queue
from decimal import Decimal
from collections import deque

import math
import sys
import csv

import random




class scale:
    
    def __init__(self, sweep = False, save = False):
        self.sweep = sweep
        self.save_space = save

    def parse_config(self):
        
        content_path = os.getcwd()
        
        config = content_path + "/Configuration Files/"+ sys.argv[1]+'.cfg'

        #topo_1 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[2]+'_profile.csv'
        #topo_2 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[3]+'_profile.csv'
        #topo_3 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[4]+'_profile.csv'
        #topo_4 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[5]+'_profile.csv'

        #topo_1 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[2]+'_profile_hbm.csv'
        #topo_2 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[3]+'_profile_hbm.csv'
        #topo_3 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[4]+'_profile_hbm.csv'
        #topo_4 = content_path + "/Profiles/google_edge_tpu/"+ sys.argv[5]+'_profile_hbm.csv'

        profile_1 = content_path + "/Profiles/google_edge_tpu_4x/"+ sys.argv[2]+'_profile_hbm.csv'
        profile_2 = content_path + "/Profiles/google_edge_tpu_4x/"+ sys.argv[3]+'_profile_hbm.csv'
        profile_3 = content_path + "/Profiles/google_edge_tpu_4x/"+ sys.argv[4]+'_profile_hbm.csv'
        profile_4 = content_path + "/Profiles/google_edge_tpu_4x/"+ sys.argv[5]+'_profile_hbm.csv'

        #sys.stdout = open(content_path+"/"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]+".txt", "w")
        
        
        #print(config)
        #print(profile_1)
        #print(profile_2)
        #print(profile_3)
        #print(profile_4)

        general = 'general'
        arch_sec = 'architecture_presets'
        net_sec  = 'network_presets'
      
       
        config_filename = config
        print("Using Architechture from ",config_filename)

        config = cp.ConfigParser()
        config.read(config_filename)

        ## Read the run name
        self.run_name = config.get(general, 'run_name')

        ## Read the architecture_presets
        ## Array height min, max
        ar_h = config.get(arch_sec, 'ArrayHeight').split(',')
        self.ar_h_min = ar_h[0].strip()

        if len(ar_h) > 1:
            self.ar_h_max = ar_h[1].strip()
        
        ## Array width min, max
        ar_w = config.get(arch_sec, 'ArrayWidth').split(',')
        self.ar_w_min = ar_w[0].strip()

        if len(ar_w) > 1:
            self.ar_w_max = ar_w[1].strip()

        ## IFMAP SRAM buffer min, max
        ifmap_sram = config.get(arch_sec, 'IfmapSramSzkB').split(',')
        self.isram_min = ifmap_sram[0].strip()

        if len(ifmap_sram) > 1:
            self.isram_max = ifmap_sram[1].strip()


        ## FILTER SRAM buffer min, max
        filter_sram = config.get(arch_sec, 'FilterSramSzkB').split(',')
        self.fsram_min = filter_sram[0].strip()

        if len(filter_sram) > 1:
            self.fsram_max = filter_sram[1].strip()


        # OFMAP SRAM buffer min, max
        ofmap_sram = config.get(arch_sec, 'OfmapSramSzkB').split(',')
        self.osram_min = ofmap_sram[0].strip()

        if len(ofmap_sram) > 1:
            self.osram_max = ofmap_sram[1].strip()

        self.dataflow= config.get(arch_sec, 'Dataflow')

        iofmap_offset = config.get(arch_sec, 'IfmapOffset')
        self.iofmap_offset = int(iofmap_offset.strip())

        filter_offset = config.get(arch_sec, 'FilterOffset')
        self.filter_offset = int(filter_offset.strip())

 
        self.profile_1= profile_1
        self.profile_2= profile_2
        self.profile_3= profile_3
        self.profile_4= profile_4



    def run_scale(self):
       
        self.parse_config()

        if self.sweep == False:
            underutilization=self.run_once()
        else:
            self.run_sweep()

        return  underutilization

    def run_once(self):

        df_string = "Output Stationary"
        if self.dataflow == 'ws':
            df_string = "Weight Stationary"
        elif self.dataflow == 'is':
            df_string = "Input Stationary"

        print("====================================================")
        print("******************* SCALE SIM **********************")
        print("====================================================")
        print("Array Size: \t" + str(self.ar_h_min) + "x" + str(self.ar_w_min))
        print("SRAM IOFMAP: \t" + str(self.isram_min+self.osram_min))
        print("SRAM Filter: \t" + str(self.fsram_min))
       
        print("CSV file path: \t" + self.profile_1 +"\n"+self.profile_2+"\n"+self.profile_3+"\t"+self.profile_4)
        print("Dataflow: \t" + df_string)
        print("====================================================")

        self.net_name_1 = self.profile_1.split('/')[-1].split('.')[0]
        self.net_name_2 = self.profile_2.split('/')[-1].split('.')[0]
        self.net_name_3 = self.profile_3.split('/')[-1].split('.')[0]
        self.net_name_4 = self.profile_4.split('/')[-1].split('.')[0]
        
        self.offset_list = [self.iofmap_offset, self.filter_offset]

        underutilization=run_nets(    word_size_bytes=2,
        			block_size=1024,
                    ifmap_sram_size=int(self.isram_min),
                    filter_sram_size = int(self.fsram_min),
                    ofmap_sram_size  = int(self.osram_min),
                    array_h = int(self.ar_h_min),
                    array_w = int(self.ar_w_min),
                    net_name_1 = self.net_name_1,
                    net_name_2 = self.net_name_2,
                    net_name_3 = self.net_name_3,
                    net_name_4 = self.net_name_4,
                    data_flow = self.dataflow,
                    profile_1 = self.profile_1,
                    profile_2 = self.profile_2,
                    profile_3 = self.profile_3,
                    profile_4 = self.profile_4,
                    offset_list = self.offset_list,
                )
        
        print("************ SCALE SIM Run Complete ****************")
        return underutilization
    
    def cleanup(self):
        if not os.path.exists("./outputs/"):
            os.system("mkdir ./outputs")

        net_name_1 = self.profile_1.split('/')[-1].split('.')[0]
        net_name_2 = self.profile_2.split('/')[-1].split('.')[0]
        net_name_3 = self.profile_3.split('/')[-1].split('.')[0]
        net_name_4 = self.profile_4.split('/')[-1].split('.')[0]

        path = "./output/scale_out"
        if self.run_name == "":
            path = "./outputs/" + net_name_1 + net_name_2 + net_name_3+ net_name_4 +"_"+ self.dataflow
        else:
            path = "./outputs/" + self.run_name

        if not os.path.exists(path):
            os.system("mkdir " + path)
        else:
            t = time.time()
            new_path= path + "_" + str(t)
            os.system("mv " + path + " " + new_path)
            os.system("mkdir " + path)


        cmd = "mv *.csv " + path
        os.system(cmd)

        cmd = "mkdir " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*sram* " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*dram* " + path +"/layer_wise"
        os.system(cmd)

        if self.save_space == True:
            cmd = "rm -rf " + path +"/layer_wise"
            os.system(cmd)



def run_nets( block_size=None,
			 ifmap_sram_size=256,
             filter_sram_size=128,
             ofmap_sram_size=1,
             array_h=32,
             array_w=32,
             data_flow = 'os',
             profile_1 = './topologies/yolo_v2.csv',
             net_name_1='yolo_v2',
             profile_2 = './topologies/yolo_v2.csv',
             net_name_2='yolo_v2',
             profile_3 = './topologies/yolo_v2.csv',
             net_name_3='yolo_v2',
             profile_4 = './topologies/yolo_v2.csv',
             net_name_4='yolo_v2',
             offset_list = [0, 10000000],
             default_read_bw=64, #64 for HBM and 8 for DDR, use HBM as AIMT
             word_size_bytes=None,
             #Balancing memory bandwidth and compute resources
             number_of_subarrays=256, #AIMT has 16 128x128 subarrays for iso resource comparison we need 256, 32x32 subarrays (262144 PEs)
             number_of_memory_channels=16# Similar to AIMT
            ):

    first = True
        
    #Arrival times of inferences are Gausian/ Normally Distributed
    mean = 0
    sigma = 1

    random.seed(0)
    arr_time_1=0
    #arr_time_2=1000*abs(random.gauss(mean, sigma))
    #arr_time_3=1000*abs(random.gauss(mean, sigma))
    #arr_time_4=1000*abs(random.gauss(mean, sigma))
    arr_time_2=0
    arr_time_3=0
    arr_time_4=0

    global arr_time

    arr_time=[arr_time_1,arr_time_2,arr_time_3,arr_time_4]

    global start
    start=0

    #Workload Specification
    #Let the application decide the batch size or go for the maximum batch size
    batch_size_1= int(sys.argv[6])
    batch_size_2= int(sys.argv[7])
    batch_size_3= int(sys.argv[8])
    batch_size_4= int(sys.argv[9])


    #DRAM Accesses
    totalDRAM=0 
    totalSRAM=0
    totalMAC=0


    global inf_m_done
    inf_m_done=[0,0,0,0]
 
    global inf_c_done
    inf_c_done=[0,0,0,0]    

    #Amount of remaining memory
    global rem_memory
    rem_memory= int((ifmap_sram_size+filter_sram_size+ofmap_sram_size)*1024/block_size)

    #Global variables for AI-MT Scheduler

    #RM-C is required MB cycles to fill the remaining SRAM capacity
    global RM_C
    RM_C = int((ifmap_sram_size+ ofmap_sram_size +filter_sram_size)*1024/(word_size_bytes*default_read_bw*number_of_memory_channels))  
    

    global threshold
    threshold=0
    global AVL_CB
    AVL_CB=0
    global MB_C
    MB_C=0
    global CB_C
    CB_C=0
    global clock
    clock=0
    global target_CB
    target_CB=None
    global target_MB
    target_MB=None  
    global m_stall
    m_stall=0
    global c_stall
    c_stall=0

    global mini_batching_stack_0
    mini_batching_stack_0=[]
    global mini_batching_stack_1
    mini_batching_stack_1=[]
    global mini_batching_stack_2
    mini_batching_stack_2=[]
    global mini_batching_stack_3
    mini_batching_stack_3=[]
    
     

    #for row in param_file:
    global MB_progress
    MB_progress=[1, 1, 1, 1]
    global CB_progress
    CB_progress=[1, 1, 1, 1]
    
    global MB_CQ
    MB_CQ = []
    global MB_CQ_DNN
    MB_CQ_DNN=[]
    global MB_CQ_BSIZE
    MB_CQ_BSIZE=[]

    global CB_CQ
    CB_CQ=[]
    global CB_CQ_DNN
    CB_CQ_DNN=[]
    global CB_CQ_BSIZE
    CB_CQ_BSIZE=[]

    global CB_SQ
    CB_SQ=[]
    global CB_SQ_DNN
    CB_SQ_DNN=[]
    global CB_SQ_BSIZE
    CB_SQ_BSIZE=[]

    global MB_inf
    MB_inf=0
    global CB_inf
    CB_inf=0
    global batch_size
    
    batch_size=[batch_size_1,batch_size_2,batch_size_3,batch_size_4]

    #rows hold the rows from meta data table, eg: rows[0]<= DNN1
    
    global rows
    rows = [[0]*12]*4

    with open(profile_1) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows_1 = list(csv_reader)
        rows[0]=rows_1

    with open(profile_2) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows_2 = list(csv_reader)
        rows[1]=rows_2

    with open(profile_3) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows_3 = list(csv_reader)
        rows[2]=rows_3

    with open(profile_4) as csv_file:
        csv_reader = csv.reader(csv_file)
        rows_4 = list(csv_reader)
        rows[3]=rows_4

    #MB_cycles=prev_MB_finish( number_of_subarrays= number_of_subarrays,number_of_memory_channels=number_of_memory_channels)
    
    clock=0
    prev_clock=0
    c_stall_cycles=0
    m_stall_cycles=0
    delta=0
    
    global MB_Count
    MB_Count=0

    global CB_Count
    CB_Count=0

    c_stall_cycles=0
    m_stall_cycles=0

    first=0

    MB_cycles=0
    CB_cycles=0

    remm=0
    remc=0

    global queue_lock

    queue_lock=0

    while(1):
        

        
        clock+=min(remc,remm)
        

        
        
        if((remm-(clock-prev_clock))==0 or (remm-(clock-prev_clock))==clock):
            if (queue_lock==0):
                MB_cycles=prev_MB_finish( number_of_subarrays= number_of_subarrays,number_of_memory_channels=number_of_memory_channels)
            else:
                MB_cycle=merge_batches_MB(number_of_subarrays= number_of_subarrays,number_of_memory_channels=number_of_memory_channels)

        elif(CB_cycles!=0):
            MB_cycles=MB_cycles-(clock-prev_clock)

        else:
            MB_cycles=0
        
        if(first!=0 and ((remc-(clock-prev_clock))==0 or (remc-(clock-prev_clock))==clock)):
            if (queue_lock==0):
                CB_cycles=prev_CB_finish( number_of_subarrays=number_of_subarrays,number_of_memory_channels=number_of_memory_channels)
            else:
                CB_cycle=merge_batches_CB(number_of_subarrays= number_of_subarrays,number_of_memory_channels=number_of_memory_channels)
        
        elif(MB_cycles!=0):
            CB_cycles=CB_cycles-(clock-prev_clock)

        else:
            CB_cycles=0
        first=1

        #print('MB_cycles', MB_cycles)
        #print('CB_cycles', CB_cycles)

        remc= CB_cycles
        remm= MB_cycles  

   
        

        if (CB_cycles==0):
            clock=clock+MB_cycles
            c_stall_cycles=c_stall_cycles+remm

        if (MB_cycles==0):
            clock=clock+CB_cycles
            m_stall_cycles=m_stall_cycles+remc

 
        #print('prev_clk', prev_clock)
        prev_clock=clock
    
 

        print('MB Progress',MB_progress)
        print('CB Progress',CB_progress)

        print(rem_memory)

        print('clock',clock)
        print('m_stall_cycles',m_stall_cycles)
        print('c_stall_cycles',c_stall_cycles)
 
        print('MB_CQ:',MB_CQ)

        print('CB_CQ:',CB_CQ)

        print('BS0',mini_batching_stack_0)
        print('BS1',mini_batching_stack_1)
        print('BS2',mini_batching_stack_2)
        print('BS3',mini_batching_stack_3)

        print('************************************************')



        if(inf_c_done[0]==1 and inf_c_done[1]==1 and inf_c_done[2]==1 and inf_c_done[3]==1):
                break


    results_file_1 = open('results_baseline.csv', 'a')
    results_file_1.write(str(sys.argv[2])+",\t"+ str(sys.argv[3])+",\t"+str(sys.argv[4])+",\t"+str(sys.argv[5])+",\t"+str(clock)+",\t"+str(m_stall_cycles)+",\t"+str(c_stall_cycles)+",\t"+str(m_stall_cycles/clock*100)+",\t"+str(c_stall_cycles/clock*100)+"\n")
        
    results_file_1.close() 




########################################################################################################
#Function to Check if two mini batches could be merged
def check_if_mergable(DNN,temp_layer,batch_size,previous_batch_size): 

    global rows
    global rem_memory

    lag_layer=temp_layer

    mergable=0
    
    

    while(lag_layer<=CB_progress[DNN]):
        
        #check if memory is sufficient for layers in between the current layer and the most recent layer
        memory_required= int(rows[DNN][lag_layer][8])+(int(rows[DNN][lag_layer][9])+int(rows[DNN][lag_layer][10]))*batch_size

        if (memory_required>rem_memory):
            mergable=0
            break

        else:
            mergable=1

        lag_layer+=1
    
    
    if(mergable==1 and lag_layer==CB_progress[DNN] ):
        
        #check if memory is sufficient when batches are merged 
        memory_required= int(rows[DNN][lag_layer][8])+(int(rows[DNN][lag_layer][9])+int(rows[DNN][lag_layer][10]))*(batch_size+previous_batch_size)

        if (memory_required>rem_memory):
            mergable=0

        else:
            mergable=1


    return mergable

########################################################################################################
#Function to Merge Mini Batches
def merge_batches_MB(number_of_subarrays,number_of_memory_channels):

    global lag_layer_m
    global lag_DNN
    

    MB_cycle=0

    if(lag_layer_m<MB_progress[lag_DNN]):
        MB_cycle=math.ceil(int(rows[lag_DNN][lag_layer_m][11])/number_of_memory_channels)
        lag_layer_m=lag_layer_m+1

    return MB_cycle


def merge_batches_CB(number_of_subarrays,number_of_memory_channels):
    
    global batch_size
    global lag_layer_c
    global queue_lock
    global lag_DNN
    global branch_batch_size
    

    if(lag_layer_c<CB_progress[lag_DNN]):
        if (lag_layer_m>=lag_layer_c):
            CB_cycle=math.ceil(int(rows[lag_DNN][lag_layer_c][15])*branch_batch_size/(number_of_subarrays))
            lag_layer_c=lag_layer_c+1
        else:
            CB_cycle=0
    else:
        batch_size[lag_DNN]=batch_size[lag_DNN]+branch_batch_size
        CB_cycle=0
        queue_lock=0


    return CB_cycle
    

#########################################################################################################
#Function to Calculate Local Batch Size

def determine_batch_size(prev_batch_size,target_MB):

    local_batch_size=0
    local_mem_req=0

    global MB_CQ
    global MB_CQ_DNN
   
    global rem_memory
    
    while (1):
        
        local_batch_size+=1

        local_mem_req= int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][8])+(int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][9])+int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][10]))*local_batch_size

    
        if (local_mem_req>rem_memory or local_batch_size>prev_batch_size):
            
            local_batch_size=local_batch_size-1
           
            
            break
        
    if (local_batch_size<0):
            local_batch_size=0     

    return local_batch_size  

##########################################################################################################
#Function to Insert MB into MB_CQ    
def insert_into_MB_CQ():
    
        global RM_C 
        global threshold
        global AVL_CB
        global MB_C
        global CB_C
        global clock
       
        global MB_CQ
        global MB_CQ_DNN
        global CB_CQ
        global CB_CQ_DNN
        global CB_SQ
        global rows
        global batch_size
        global inf_m_done
        global MB_progress
        global MB_inf
        global m_stall
        
        global arr_time
        global start

        global queue_lock

        global CB_CQ_BSIZE
        global MB_CQ_BSIZE

 
        while(1):

            
            if(arr_time[MB_inf]>clock and start==1):
                if(MB_inf>=3):
                    MB_inf=0
                        
                else:
                    MB_inf+=1
                        
                continue
            
            

            if (len(MB_CQ)<12):

                
                start=1

                if(len(rows[MB_inf])-1==MB_progress[MB_inf]):
                    inf_m_done[MB_inf]=1

                if(inf_m_done[MB_inf]==1):
                   
                    if(MB_inf>=3):
                        MB_inf=0
                        break
                    else:
                        MB_inf+=1
                        break
                    

                
                
                if(len(rows[MB_inf])-1>=MB_progress[MB_inf] ):
                #if(1):
                    

                    in_queue=0

                    
                    for j in range(len(MB_CQ)):
                        MB=MB_CQ[j]
                        DNN=MB_CQ_DNN[j]
                        BS=MB_CQ_BSIZE[j]
                        if (MB==MB_progress[MB_inf] and DNN==MB_inf):
 
                                in_queue=1
                                break
                               

                    if(in_queue==0):
                       
                       if (len(rows[MB_inf])-1 > MB_progress[MB_inf] and queue_lock==0):
                            MB_CQ.append(MB_progress[MB_inf])
 
                            MB_CQ_DNN.append(MB_inf)
                            MB_CQ_BSIZE.append(batch_size[MB_inf])
                            MB_progress[MB_inf]=MB_progress[MB_inf]+1
                        
                    if(MB_inf>=3):
                            MB_inf=0
                            #break
                    else:
                            MB_inf+=1
                            #break
        

            else:

                if(MB_inf>=3):
                    MB_inf=0
                    break
                else:
                    MB_inf+=1
                    break
            
           
##########################################################################################################
#Function to Insert CB into CB_CQ                 

def insert_into_CB_CQ():
        
        global RM_C 
        global threshold
        global AVL_CB
        global MB_C
        global CB_C
        global clock
       
        global MB_CQ
        global MB_CQ_DNN
        global CB_CQ
        global CB_CQ_DNN
        global CB_SQ
        global rows
        global batch_size
        global inf_m_done
        global inf_c_done
        global MB_progress
        global CB_progress
        global CB_inf 

        global rem_memory
        
        global prev_MB
        global prev_MB_DNN

        global CB_CQ_BSIZE

        global stack_lock
        stack_lock=[1,1,1,1]
        global queue_lock

        if (len(CB_CQ)<12):

                    
                if (MB_CQ):


                            
                            if(len(rows[CB_inf])-1==CB_progress[CB_inf]):
                                
                                inf_c_done[CB_inf]=1
           
                #All memory access done, just finish remaining compute
                else:
                    #CB_inf=0
                    while(1):
                        


                        if(len(rows[CB_inf])-1>CB_progress[CB_inf] and queue_lock==0):
                            CB_CQ.append(CB_progress[CB_inf])
                            CB_CQ_DNN.append(CB_inf)
                            CB_CQ_BSIZE.append(batch_size[CB_inf])
                        
                        if (CB_inf==3):
                            CB_inf=0
                            break
                        else:
                            CB_inf+=1
                            break
                


##########################################################################################################
#Trigger when previous MB finish

def prev_MB_finish( number_of_subarrays, number_of_memory_channels):
        
        insert_into_MB_CQ()
         
        global threshold
        global AVL_CB
        global MB_C
        global CB_C
        global clock
        global batch_size
        global RM_C 
       

        global MB_CQ
        global MB_CQ_DNN
        global CB_CQ
        global CB_CQ_DNN
        global CB_SQ
        global MB_CQ_BSIZE
        global CB_CQ_BSIZE
        global CB_SQ_BSIZE
        global rows
        global inf_m_done
        global MB_progress
        global CB_progress
        
        global m_stall
        global rem_memory

        global prev_MB
        global prev_MB_DNN

        global MB_Count

        global stack_lock
        global queue_lock

        ###############
        #AI-MT Sheduler
        ###############
        target_MB=None
        MB_cycle=0

        for i in range(len(MB_CQ)):
           

           
            MB=MB_CQ[i]
            DNN=MB_CQ_DNN[i]
            BSIZE=MB_CQ_BSIZE[i]


            
            MB_cycle=math.ceil(int(rows[DNN][MB][11])/number_of_memory_channels)
            
            #When your batch size is higher it consumes a propotional ammount of cycles for the computation, so cycles for batch size of 1, should be multipled by batch size not divivded
            CB_cycle=math.ceil(int(rows[DNN][MB][15])*BSIZE/(number_of_subarrays))


            mem_req= int(rows[MB_CQ_DNN[i]][MB_CQ[i]][8])+(int(rows[MB_CQ_DNN[i]][MB_CQ[i]][9])+int(rows[MB_CQ_DNN[i]][MB_CQ[i]][10]))*BSIZE
            local_batch_size=BSIZE

          
            if(rem_memory>=mem_req):
                if (MB_cycle< RM_C):
                
                #AVL_CB, resolved CBs, smaller than threshold, check if MB's MB cycle smaller than CB cycle
               
                    if (AVL_CB < threshold) :
                        if(CB_cycle>MB_cycle):
                            target_MB=i
                            
                            break
                    
                    #if avail CB is lesser
                    else:
                        
                        target_MB=i
                        
                        break
            

        if(target_MB==None and len(MB_CQ)!=0):
        #Stall until executing CB finish
        #AVL_CB decreases for executed CBs     
            RM_C+=1
            #clock+=1
            MB_cycle=0
            print('Memory Insufficient')

            #Find the candidate with minimum memory requirement    
            memory_req=0
            min_mem_req=math.inf
            min_mem_MB=None

           
            for i in range(len(MB_CQ)):


                memory_req= int(rows[MB_CQ_DNN[i]][MB_CQ[i]][8])+(int(rows[MB_CQ_DNN[i]][MB_CQ[i]][9])+int(rows[MB_CQ_DNN[i]][MB_CQ[i]][10]))*BSIZE
             
                if (min_mem_req>memory_req):
                    min_mem_req=memory_req
                    min_mem_MB=i
                     
            

            MB=MB_CQ[min_mem_MB]
            DNN=MB_CQ_DNN[min_mem_MB]
            BSIZE=MB_CQ_BSIZE[min_mem_MB]
                

                
            local_batch_size=determine_batch_size(BSIZE,min_mem_MB)
            
            

            if (local_batch_size!=0):

                MB_cycle=math.ceil(int(rows[DNN][MB][11])/number_of_memory_channels)
                CB_cycle=math.ceil(int(rows[DNN][MB][15])*local_batch_size/(number_of_subarrays))                


                if (MB_cycle< RM_C):
                    
                    #AVL_CB, resolved CBs, smaller than threshold, check if MB's MB cycle smaller than CB cycle
                
                        if (AVL_CB < threshold) :
                            if(CB_cycle>MB_cycle):
                                target_MB=min_mem_MB
                                
                                
                        
                        #if avail CB is lesser
                        else:
                            
                            target_MB=min_mem_MB

                            
            else:
                                
                    RM_C+=1
                    #clock+=1
                    MB_cycle=0
                    print('No target MB')
         


        if (target_MB!=None and queue_lock==0):
                
            if(BSIZE==local_batch_size):

                print('Memory Sufficient')
                print('MB Layer:',MB_CQ[target_MB],'MB DNN:',MB_CQ_DNN[target_MB])
                
                # Allocate memory for inputs outputs and weights
                mem_req= int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][8])+(int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][9])+int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][10]))*MB_CQ_BSIZE[target_MB]
                
                #MB[target_MB].append(MB[target_MB]+1)
                MB=MB_CQ.pop(target_MB)
                target_MB_DNN=MB_CQ_DNN.pop(target_MB)
                BSIZE=MB_CQ_BSIZE.pop(target_MB)
                
                CB_CQ.append(MB)
                CB_CQ_DNN.append(target_MB_DNN)
                CB_CQ_BSIZE.append(BSIZE)

                #if (MB_progress[target_MB_DNN]!=1):
                rem_memory=rem_memory-mem_req
            
                #print('rem_memory_after',rem_memory)
                
                prev_MB= MB
                prev_MB_DNN=target_MB_DNN
                
                MB_Count+=1
                print('MB_Count', MB_Count)

                if(len(rows[target_MB_DNN])-1==MB_progress[target_MB_DNN]):
                                
                    if(target_MB_DNN==0 and len(mini_batching_stack_0)==0):
                        inf_m_done[0]==1
                                        
                    if(target_MB_DNN==1 and len(mini_batching_stack_1)==0):
                        inf_m_done[1]==1
                                        
                    if(target_MB_DNN==2 and len(mini_batching_stack_2)==0):
                        inf_m_done[2]==1
                                        
                    if(target_MB_DNN==3 and len(mini_batching_stack_3)==0):
                        inf_m_done[3]==1
            

                MB_cycle=math.ceil(int(rows[target_MB_DNN][MB][11])/number_of_memory_channels)
                CB_cycle=math.ceil(int(rows[target_MB_DNN][MB][15])*BSIZE/(number_of_subarrays))

            
            else:

                print('Memory Insufficient')
                print('Reducing Batch Size')
                print('MB Layer:',MB_CQ[target_MB],'MB DNN:',MB_CQ_DNN[target_MB])

                local_mem_req= int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][8])+(int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][9])+int(rows[MB_CQ_DNN[target_MB]][MB_CQ[target_MB]][10]))*local_batch_size
                rem_work=BSIZE-local_batch_size

                MB=MB_CQ.pop(target_MB)
                target_MB_DNN=MB_CQ_DNN.pop(target_MB)
                MB_CQ_BSIZE.pop(target_MB)

                CB_CQ.append(MB)
                CB_CQ_DNN.append(target_MB_DNN)
                CB_CQ_BSIZE.append(local_batch_size)
                

                rem_memory=rem_memory-local_mem_req
                
                if(target_MB_DNN==0):
                    mini_batching_stack_0.append((MB,rem_work))
                    batch_size[0]=local_batch_size
                    print('BS:',mini_batching_stack_0)
                if(target_MB_DNN==1):
                    mini_batching_stack_1.append((MB,rem_work)) 
                    batch_size[1]=local_batch_size
                    print('BS:',mini_batching_stack_1)
                if(target_MB_DNN==2):
                    mini_batching_stack_2.append((MB,rem_work))
                    batch_size[1]=local_batch_size
                    print('BS:',mini_batching_stack_2)
                if(target_MB_DNN==3):
                    mini_batching_stack_3.append((MB,rem_work))
                    batch_size[1]=local_batch_size
                    print('BS:',mini_batching_stack_3)
                
        
                MB_Count+=1
                print('MB_Count', MB_Count)

                if(len(rows[target_MB_DNN])-1==MB_progress[target_MB_DNN]):
                                
                    if(target_MB_DNN==0 and len(mini_batching_stack_0)==0):
                        inf_m_done[0]==1
                                        
                    if(target_MB_DNN==1 and len(mini_batching_stack_1)==0):
                        inf_m_done[1]==1
                                        
                    if(target_MB_DNN==2 and len(mini_batching_stack_2)==0):
                        inf_m_done[2]==1
                                        
                    if(target_MB_DNN==3 and len(mini_batching_stack_3)==0):
                        inf_m_done[3]==1
            

                MB_cycle=math.ceil(int(rows[target_MB_DNN][MB][11])/number_of_memory_channels)
                CB_cycle=math.ceil(int(rows[target_MB_DNN][MB][15])*local_batch_size/(number_of_subarrays))



            MB_C=MB_C+MB_cycle
            
            RM_C=RM_C-MB_cycle
            
            
            AVL_CB=max(AVL_CB-MB_cycle,0)+CB_cycle

        
        if (queue_lock==0):
            
            for i, CB_x in enumerate(CB_CQ):

                                

                DNN=CB_CQ_DNN[i]

                CB=CB_CQ[i]
                BS=CB_CQ_BSIZE[i]

                CB_cycle=math.ceil(int(rows[DNN][CB][15])*BS/(number_of_subarrays))

               
                if (CB_C<MB_C and MB_progress[DNN]>=CB):
                    
                    CB_C+=CB_cycle
                    CB_SQ.append(CB_CQ.pop(i))
                    CB_SQ_DNN.append(CB_CQ_DNN.pop(i))
                    CB_SQ_BSIZE.append(CB_CQ_BSIZE.pop(i))
                

                else:
                    #CB cycles decrease at each clock when memory is stalled and computation allowed to move forward
                    CB_C=CB_C-1
                

            #If memory access is complete just finish the remaining compute
            if(len(MB_CQ)==0):
                
                if (len(CB_CQ)!=0):
                    CB_SQ.append(CB_CQ.pop(0))
                    CB_SQ_DNN.append(CB_CQ_DNN.pop(0))
                    CB_SQ_BSIZE.append(CB_CQ_BSIZE.pop(0))

                    MB_cycle=0


        return MB_cycle
            

##########################################################################################################
#Trigger when previous CB finish

def prev_CB_finish(number_of_subarrays,number_of_memory_channels):
        
        global threshold
        global AVL_CB
        global MB_C
        global CB_C
        global clock
        global batch_size
        global RM_C 
        

        global MB_CQ
        global MB_CQ_DNN
        global CB_CQ
        global CB_CQ_DNN
        global CB_SQ
        global rows
        global inf_m_done
        global inf_c_done
        global MB_progress
        global CB_progress
        global rem_memory

        global c_stall

        CB_cycle=0
        
        global CB_Count

        global stack_lock

        global lag_layer_m
        global lag_layer_c
       
        global queue_lock
       
        global lag_DNN
        global branch_batch_size

        queue_lock=0

        ###############################################################
            #If stack is unlocked release left behind work from stack
            #(MB,rem_work)

        if (len(mini_batching_stack_0)!=0):
                data= mini_batching_stack_0.pop()
                temp_layer=data[0]
                rem_work=data[1]
                if(check_if_mergable(0,temp_layer,rem_work,batch_size[0]))==0:
                    mini_batching_stack_0.append(data)
                else:
                    queue_lock=1
                    lag_layer_m=temp_layer
                    lag_layer_c=temp_layer
                    lag_DNN=0
                    branch_batch_size=rem_work
                

        if (len(mini_batching_stack_1)!=0):
                data= mini_batching_stack_1.pop()
                temp_layer=data[0]
                rem_work=data[1]
                if(check_if_mergable(1,temp_layer,rem_work,batch_size[1]))==0:
                    mini_batching_stack_1.append(data)
                else:
                    queue_lock=1
                    lag_layer_m=temp_layer
                    lag_layer_c=temp_layer
                    lag_DNN=1
                    branch_batch_size=rem_work

        if (len(mini_batching_stack_2)!=0):
                data= mini_batching_stack_2.pop()
                temp_layer=data[0]
                rem_work=data[1]
                if(check_if_mergable(2,temp_layer,rem_work,batch_size[2]))==0:
                    mini_batching_stack_2.append(data)
                else:
                    queue_lock=1
                    lag_layer_m=temp_layer
                    lag_layer_c=temp_layer
                    lag_DNN=2
                    branch_batch_size=rem_work
            
        if (len(mini_batching_stack_3)!=0):
                data= mini_batching_stack_3.pop()
                temp_layer=data[0]
                rem_work=data[1]
                if(check_if_mergable(3,temp_layer,rem_work,batch_size[3]))==0:
                    mini_batching_stack_3.append(data)
                else:
                    queue_lock=1
                    lag_layer_m=temp_layer
                    lag_layer_c=temp_layer
                    lag_DNN=3
                    branch_batch_size=rem_work

        ##############################################################

        print('rem_memory',rem_memory)

        insert_into_CB_CQ()

        
 
        CB_SQ_copy= CB_SQ
        if (len(CB_SQ_copy)!=0 and CB_progress[CB_SQ_DNN[0]]+1<=MB_progress[CB_SQ_DNN[0]] and queue_lock==0):

            
            c_stall=0

            target=CB_SQ.pop(0)
            target_DNN=CB_SQ_DNN.pop(0)
            target_BSIZE=CB_SQ_BSIZE.pop(0)

        

            CB_progress[target_DNN]+=1     
            batch_size[target_DNN]=target_BSIZE

            CB_cycle=math.ceil(int(rows[target_DNN][target][15])*target_BSIZE/(number_of_subarrays))

            

            RM_C=RM_C+math.ceil(int(rows[target_DNN][target][11])/number_of_memory_channels)
            
            
            print('CB Layer:', target, 'DNN:', target_DNN)
            
            mem_freed= int(rows[target_DNN][target][8])+(int(rows[target_DNN][target][9])+int(rows[target_DNN][target][10]))*target_BSIZE

            rem_memory=rem_memory+mem_freed
            CB_Count+=1
            print('CB_Count',CB_Count)
            
            print('rem_memory_after',rem_memory)
            


        
        #If memory access is complete just finish the remaining compute
        else:
                c_stall=1 
                CB_cycle=0 
                                

        ####################################################################
        # Done only if stack is empty
        ####################################################################

        if(CB_progress[0]==len(rows[0])-1 and len(mini_batching_stack_0)==0):
                    inf_c_done[0]=1
                                        
        if(CB_progress[1]==len(rows[1])-1 and len(mini_batching_stack_1)==0):
                    inf_c_done[1]=1
                                        
        if(CB_progress[2]==len(rows[2])-1 and len(mini_batching_stack_2)==0):
                    inf_c_done[2]=1
                                        
        if(CB_progress[3]==len(rows[3])-1 and len(mini_batching_stack_3)==0):
                    inf_c_done[3]=1  


        return CB_cycle
    
scale_1= scale()
scale_1.run_scale()


