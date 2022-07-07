import os

def main():
    
    ROOT = os.getcwd()
    OUTDIR = os.path.join(ROOT,'sampling')
    
    video_format = ('.mp4','.avi','.MOV')
    class_names = []
    
    for video_file in os.listdir(os.path.join(ROOT,'video')):
        
        #Onlt video files have this format can go through
        if video_file.endswith(video_format):
            
            #trim video format
            class_name = video_file[:-4].lower()
            folder_path = os.path.join(OUTDIR,class_name)
            video_path = os.path.join(os.path.join(ROOT,'video'),video_file)
            
            #Make Directories
            try:
                os.makedirs(os.path.join(folder_path,'images'))
                os.makedirs(os.path.join(folder_path,'labels'))
            except OSError as error:
                print(error)  
            class_names.append("'" + class_name + "'")
                
            print("Directory '%s' created" %class_name)
            
            #Sampling images from video with ffmpeg
            os.system('ffmpeg -i {} -vf  fps=10 {}/img%05d.jpg'.format(video_path , os.path.join(folder_path,'images')))
            print(" '%s' has been sampling" %class_name)
        
    #Create class_name.txt file 
    with open(os.path.join(ROOT,'class_name.txt'),'w') as f:
        f.write('[')
        f.write(",".join(class_names))
        f.write(']')
        
if __name__ == '__main__':
    main()