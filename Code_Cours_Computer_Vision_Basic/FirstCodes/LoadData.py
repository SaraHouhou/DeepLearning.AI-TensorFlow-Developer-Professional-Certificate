import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def dataloadAndShow(base_dir) :
 #base_dir = 'Data/'

    print("Contents of base directory:")
    print(os.listdir(base_dir))

    train_dir = os.path.join(base_dir, 'training')
    validation_dir = os.path.join(base_dir, 'validation')

    print("\nContents of train directory:")
    print(os.listdir(train_dir))

    print("\nContents of validation directory:")
    print(os.listdir(validation_dir))

    # Directory with training cat/dog pictures
    train_call_dir = os.path.join(train_dir, 'call')
    train_menu_dir = os.path.join(train_dir, 'menu')

    # Directory with validation cat/dog pictures
    validation_call_dir = os.path.join(validation_dir, 'call')
    validation_menu_dir = os.path.join(validation_dir, 'menu')

    train_call_fnames = os.listdir( train_call_dir )
    train_menu_fnames = os.listdir( train_menu_dir )

    print(train_call_fnames[:10])
    print(train_menu_fnames[:10])



    #let's find out the total number of call and menu images in the train and validation directories:
        
        
    print('total training call images :', len(os.listdir(train_call_dir ) ))
    print('total validation call images :', len(os.listdir(validation_call_dir ) ))

    print(' total training menu images :', len(os.listdir(train_menu_dir) ))
    print('total validation menu images :', len(os.listdir(validation_menu_dir ) ))

    #take a look at a few pictures to get a better sense of what the call and menu datasets look like. First, configure the matplotlib parameters:
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4

    pic_index = 0 # Index for iterating over images

    #Display a batch of 8 call and 8 menu pictures. You can re-run the cell to see a fresh batch each time:
        
        
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index+=8

    next_call_pix = [os.path.join(train_call_dir, fname) 
                    for fname in train_call_fnames[ pic_index-8:pic_index] 
                    ]

    next_menu_pix = [os.path.join(train_menu_dir, fname) 
                    for fname in train_menu_fnames[ pic_index-8:pic_index]
                    ]

    for i, img_path in enumerate(next_call_pix+next_menu_pix):
    # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        #print(img.shape)

        plt.show()
