import os

def main():
    ROOT = os.getcwd()
    OUTDIR = os.path.join(ROOT,'sampling')
    
    video_format = ('.mp4','.avi','.MOV')
    class_names = []
    for video_file in os.listdir(os.path.join(ROOT,'video')):
        if video_file.endswith(video_format):
            class_name = video_file[:-4].capitalize()
            folder_path = os.path.join(OUTDIR,class_name)
            video_path = os.path.join(os.path.join(ROOT,'video'),video_file)
            try:
                os.makedirs(os.path.join(folder_path,'images'))
                os.makedirs(os.path.join(folder_path,'labels'))
            except OSError as error:
                print(error)  
            class_names.append("'" + class_name + "'")
                
            print("Directory '%s' created" %class_name)
            
            os.system('ffmpeg -i {} -vf  fps=30 {}/img%05d.jpg'.format(video_path , os.path.join(folder_path,'images')))
            print(" '%s' has been sampling" %class_name)
        
    with open(os.path.join(ROOT,'class_name.txt'),'w') as f:
        f.write('[')
        f.write(",".join(class_names))
        f.write(']')
        
if __name__ == '__main__':
    main()