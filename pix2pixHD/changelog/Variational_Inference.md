# Variational Inference - Confidence Map for Predictions

#### Why?
Model is able to generate a single prediction but we wanted to understand the variability in its predictions as a measure of confidence. High variance is a good proxy for low confidence and vic versa.
 
By introducing Dropout at inference we are able to generate N predictions for a single input. This produces an [X,Y,N] array stack of predictions which we can iterate through and also take the variance in the N axis to give a 2D variance map
 
This is to be performed in addition to regular inference using test_16bit.py
#### What?

Creation of variational_inference.py 


Added new TestOptions
1. Number of runs per image 
2. Path to save images
3. use dropout fixed at 0.2

    
        self.parser.add_argument("--variational_inf_runs",
                                 type=int, default=0,
                                 help="no. runs for variational_inference.py")
        self.parser.add_argument("--variational_inf_path", type=str,
                                 help="path to save variational inf outputs")
        self.parser.add_argument("--dropout_variation_inf",
                         choices=('True','False'),default='False',
                         help=" default usage is False but if True then"
                              "turning dropout of 0.2 on for variation"
                              "inference")

models/networks.py

1. Import new TestOptions


    ###############################################################################
    # Variational Inference
    root_dir = '../'
    sys.path.append(root_dir)
    from options.test_options import TestOptions
    opt = TestOptions().parse(save=False)
    
    if opt.dropout_variation_inf == 'False':
        dropout_variation_inf = False
    else:
        dropout_variation_inf = True
    ###############################################################################


2. Embed Options


    Line 215
        changed default use_dropout=False to use_dropout=dropout_variation_inf
        
3. Fix dropout to 0.2

   
    Line 234        
        if use_dropout:
            # conv_block += [nn.Dropout(0.5)]
            conv_block += [nn.Dropout(0.2)]

    
    