#here we will get a library. we will first group the images. then prune out cells that are out of focus. then get the background.
#then get per cell flourecence levels, and report mean / std.
# also report for each image, how many cells in total, and how many were out of focus

import CSVTable
import csv,math
import argparse
import glob,os

def get_image_groups(dir_name,query):
    groups=[]
    #each group has three files:__phase_analysis.csv, __phase_lap_analysis.csv, __signal_analysis.csv
    for phase_analysis_fn in glob.glob(dir_name+"/*%s*__phase_analysis.csv"%query):
        #make sure the other files are also there
        phase_lap_fn=phase_analysis_fn.split("__phase_analysis.csv")[0]+"__phase_lap_analysis.csv"
        signal_fn=phase_analysis_fn.split("__phase_analysis.csv")[0]+"__signal_analysis.csv"
        if not os.path.isfile(phase_lap_fn):
            raise (Exception("can not find file %s"%phase_lap_fn))
        if not os.path.isfile(signal_fn):
            raise (Excpetion("can not find file %s"%signal_fn))
        groups.append([phase_analysis_fn,phase_lap_fn,signal_fn])
    return groups

#sample_row=[image_id,Label,Area,Mean,StdDev,Mode,Min,Max,XM,YM,Circ.,IntDen,,Median,RawIntDen,	AR,Round,	Solidity]

def analyze_group(group,group_name,cur,sample_data):
    [phase_analysis_fn,phase_lap_fn,signal_fn]=group
    #load phase_lap_fn 
    table_name=phase_analysis_fn.split("/")[-1].split(".csv")[0].replace("-","_")
    handler=open(phase_lap_fn,'rb')
    tables.add_table(table_name,handler)
    #add signal data
    table_name2=signal_fn.split("/")[-1].split(".csv")[0].replace("-","_")
    handler=open(signal_fn,'rb')
    tables.add_table(table_name2,handler)
    #### get some statistics
    cur.execute('SELECT Mean from '+table_name) #select all cells
    all_rows=cur.fetchall()
    all_vals=[]
    focus_vals=[]
    for e in all_rows:
        v=float(e[0])
        all_vals.append(v)
        if v>=1000:
            focus_vals.append(v)
    print "LOG: for %s %d out of %d cells are in focus. The intensity of the cells in focus ranges from %f to %f and the intensity for all cells ranges from %f to %f"%(group_name,len(focus_vals),len(all_vals),min(focus_vals),max(focus_vals),min(all_vals),max(all_vals))
    #### end statistics
    cur.execute('SELECT Label,Mean from '+table_name) #select cells in focus
    lap_rows=cur.fetchall()
    
    for l_row in lap_rows:
        if float(l_row[1])<1000:
            continue
        label='signal:'+l_row[0].split(":")[1]
        cur.execute('SELECT * from '+table_name2 + ' where Label == "%s"'%label)
        signal_row=cur.fetchall()
        row_to_add=[group_name,str(signal_row[0][1])]
        for e in signal_row[0][2:]:
            row_to_add.append(float(e))
        sample_data.append(row_to_add)
    return sample_data
    

def meanstdv(x):
    n, mean, std = len(x), 0, 0
    for a in x:
        mean = mean + a
    mean = mean / float(n)
    for a in x:
        std = std + (a - mean)**2
    std = math.sqrt(std / float(n-1))
    return mean, std

def get_average_singal(sample_data):
    avg_col=[]
    for row in sample_data[1:]:
        avg_col.append(float(row[3]))
    return meanstdv(avg_col)



def run_analysis(cur,header,input_dir_name,background_query, signal_query, output_csv_file):
    tables=CSVTable.CSVTable()
    cur=tables.create_database()
    header=["ImageId","Label","Area","Mean","StdDev","Mode","Min","Max","XM","YM","Circ.","IntDen","Median","RawIntDen","AR","Round","Solidity"]
    
    #### get background data
    groups=get_image_groups(input_dir_name,background_query)
    if len(groups) == 0 :
        print "WARNING: no images to process for query %s. Check your input directory"%(args.background_query)
        sys.exit(1)
    print "\nGoing to process",len(groups),"sets of images"
    background_data=[header]
    for group in groups:
        background_data=analyze_group(group,group[0].split("/")[-1].split(".csv")[0],cur,background_data)
    #get pixel background correction 
    correction = get_average_singal(background_data)

    print "correction is",correction[0],"with std",correction[1]


    #### get signal data
    groups=get_image_groups(input_dir_name,signal_query)
    if len(groups) == 0 :
        print "WARNING: no images to process for query %s. Check your input directory"%signal_query
        sys.exit(1)
    print "\nGoing to process",len(groups),"sets of images"
    signal_data=[header]
    for group in groups:
        signal_data=analyze_group(group,group[0].split("/")[-1].split(".csv")[0],cur,signal_data)
    #background correct the signal data
    corrected_signal_data=[header]
    for i in range(1,len(signal_data)):
        #correct the mean
        signal_data[i][3]=signal_data[i][3]-correction[0]
        if signal_data[i][3]<0:
            #this means that this cell is below background. remove it.
            print "not going to include cell",signal_data[i][1],"signal below background"
            continue
        #correct Std
        #TODO
        #correct the IntDen
        signal_data[i][11]=signal_data[i][11]-(correction[0]*signal_data[i][2])
        #correct the median
        signal_data[i][12]=signal_data[i][12]-correction[0]
        #correct RawIntDen
        signal_data[i][13]=signal_data[i][13]-(correction[0]*signal_data[i][2])
        corrected_signal_data.append(signal_data[i])
    #write out the corrected signal data
    output_f = open(output_csv_file, 'wb')
    output_writer = csv.writer(output_f)
    output_writer.writerows(corrected_signal_data)
    output_f.close()




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Analyze imageJ results for positioning project')
    parser.add_argument("-i", "--input", help="name of input directory including all imageJ results as csv files",required=True)
    parser.add_argument("-o", "--output",help="output csv file",required=True)
    parser.add_argument("-b","--background_query",help="backgroup data for background correction",required=True)
    parser.add_argument("-s","--signal_query",help="query on the input folder",required=True)
    args = parser.parse_args()
    input_dir_name=args.input
    output_csv_file=args.output
    run_analysis(cur,header,input_dir_name,args.background_query, args.signal_query, output_csv_file)

