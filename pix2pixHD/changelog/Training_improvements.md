# Training with Validation Performance Measured by SSIM & PSNR
my_aligned_dataset.py

    Line 20
    if opt.isTrain or opt.phase == "val":

train_16bit.py

    ## Add imports
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import json

    ## Add Val Dataloader and update train dataloader

    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    #dataset_size = len(data_loader)
    #print('#training images = %d' % dataset_size)

    # Load Train Set for input into model
    data_loader = CreateDataLoader(opt)
    dataset_train = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # Load Val Set
    opt.phase = "val"
    print(opt.phase)
    data_loader = CreateDataLoader(opt)
    dataset_val = data_loader.load_data()
    val_size = len(data_loader)
    print('#validation images = %d' % val_size)
    opt.phase = "train"


    ## Update for loop
    # for i, data in enumerate(dataset, start=epoch_iter):
     for i, data in enumerate(dataset_train, start=epoch_iter):

    ## Add lists
    ssim_scores, psnr_scores = [], []

    ## Update forward pass to always generate image
    losses, generated = model(Variable(data['label']), Variable(data['inst']),
            Variable(data['image']), Variable(data['feat']), infer=True) #  infer=save_fake)

    ## After model save for epoch add the following
    # Compute metric
        gen_image = util.tensor2im(generated.data[0])
        gt_image = util.tensor2im(data['image'][0])
        #if opt.val_metric == "ssim":
        score_ssim = ssim(gt_image, gen_image)
        ssim_scores.append(score_ssim)
        # elif opt.val_metric == "psnr":
        score_psnr = psnr(gt_image, gen_image)
        psnr_scores.append(score_psnr)
        if epoch_iter >= dataset_size:
            break
    avg_ssim_train = ssim_scores.mean()
    print("Averagae SSIM for Epoch {} Train = {}".format(epoch, avg_ssim_train))
    avg_psnr_train = psnr_scores.mean()
    print("Averagae PSNR for Epoch {} Val = {}".format(epoch, avg_psnr_train))

     ### Compare with validation set and save metrics
    ############## Forward Pass Val ######################
    ssim_scores, psnr_scores = [], []
    for val_img in dataset_val:
        _, generated = model(Variable(data['label']),
                             Variable(data['inst']),
                             Variable(data['image']),
                             Variable(data['feat']))
        gen_image = util.tensor2im(generated.data[0])
        gt_image = util.tensor2im(data['image'][0])
        #if opt.val_metric == "ssim":
        score_ssim = ssim(gt_image, gen_image)
        ssim_scores.append(score_ssim)
        #elif opt.val_metric == "psnr":
        score_psnr = psnr(gt_image, gen_image)
        psnr_scores.append(score_psnr)

    avg_ssim_val = ssim_scores.mean()
    print("Averagae SSIM for Epoch {} Val = {}".format(epoch, avg_ssim_val))
    avg_psnr_val = psnr_scores.mean()
    print("Averagae PSNR for Epoch {} Val = {}".format(epoch, avg_psnr_val))

    ## Save scores in json add the following lines
    #Save scores
    if epoch == 0:
        json_train = util.score_json_row(
            "log", epoch, "train", opt.checkpoints_dir,
            avg_ssim_train, avg_psnr_train)
        json_val = util.score_json_row(
            "log", epoch, "val", opt.checkpoints_dir,
            avg_ssim_val, avg_psnr_val)
        with open(os.path.join(opt.checkpoints_dir, "train_val_scores.txt"),
                  "w") as file:
            json.dump(json_train, file)
        with open(os.path.join(opt.checkpoints_dir, "train_val_scores.txt"),
                  "a") as file:
            json.dump(json_val, file)
    else:
        json_train = util.score_json_row(
            "log", epoch, "train", opt.checkpoints_dir,
            avg_ssim_train, avg_psnr_train)
        json_val = util.score_json_row(
            "log", epoch, "val", opt.checkpoints_dir,
            avg_ssim_val, avg_psnr_val)
        with open(os.path.join(opt.checkpoints_dir, "train_val_scores.txt"),
                  "a") as file:
            json.dump(json_train, file)
            json.dump(json_val, file)
