# Changes to test_16bit.py to enable 

#### Why?

Changes required as the backbone architecture for the Generator means that if the image shape eg (1080x1080) is not a multiple of 32 then the shape of the prediction will differ from the input signal. For 1080x1080 the original test_16bit.py produces an output of size 1088x1088.


#### What?

New hyper-parameter added to indicate reshape required

base_options.py


    self.parser.add_argument("--output_reshape", type=int,
                                  help="resize model output to this shape fixed to same for x and y")
test_16bit.py

    line 78 - 83
     # visuals = OrderedDict(
    #    [
    #        ('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
    #        ('synthesized_image', util.tensor2im(generated.data[0])),
    #   ],
    # )
    
    ## New Code
        
     if opt.output_reshape:
        prediction = util.tensor2im(generated.data[0])
        prediction = cv2.resize(
            prediction, (opt.output_reshape, opt.output_reshape),
            interpolation=cv2.INTER_LINEAR)
        input_label = util.tensor2label(data['label'][0], opt.label_nc)
        input_label = transform.resize(
            input_label, (opt.output_reshape, opt.output_reshape))
        visuals = OrderedDict(
             [
                 ('input_label', input_label),
                 ('synthesized_image', prediction),
             ],
         )
        print("Min orig {} Max orig {}".format(
            np.min(util.tensor2im(generated.data[0])),
            np.max(util.tensor2im(generated.data[0]))))
        print("Min reshape {} Max reshape {}".format(np.min(prediction),
                                                     np.max(prediction)))

    else:
        visuals = OrderedDict(
            [
                ('input_label',
                 util.tensor2label(data['label'][0], opt.label_nc)),
                ('synthesized_image', util.tensor2im(generated.data[0])),
            ],
        )

