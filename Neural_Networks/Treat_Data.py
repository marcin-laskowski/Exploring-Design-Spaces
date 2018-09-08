def treat_data(Images, Labels, Shapes):
    compt = 0
    # REMOVE WHEN Labels != 0 different to Shape
    for i in range(10000):
        X0 = (Labels[i,0,0] ==0 ).float()
        if not torch.equal( X0 +Shapes[i,0,0].float(), torch.ones(64,64)):
            Labels[i] = Labels[i-10]
            Images[i] = Images[i-10]
            Shapes[i] = Shapes[i-10]

            compt +=1

    # REMOVE WHEN Shape to small
    for i in range(10000):
        if Shapes[i,0,0].sum()  <200.0:

            Labels[i] = Labels[i-10]
            Images[i] = Images[i-10]
            Shapes[i] = Shapes[i-10]

            compt +=1
    # REMOVE WHEN STRESS TOO SMALL
    for i in range(10000):
        if Labels[i,0,0].sum()  <2000.0:
            Labels[i] = Labels[i-10]
            Images[i] = Images[i-10]
            Shapes[i] = Shapes[i-10]

            compt +=1

    # REMOVE PLUS 20 TO THE STRESS DISTRIBUTION TO WELL DIFFERENCIATE WITH THE OUTPUT
    for i in range(10000):
        Labels[i,0,0].float()+X0*20


    #list_to_shuffle = torch.randperm(10000)
    #Labels = Labels[list_to_shuffle]
    #Images = Images[list_to_shuffle]
    #Shapes = Shapes[list_to_shuffle]
    print("DATA TREATED ! {}% of the data removed".format(compt/10000*100))
    return Images, Labels, Shapes
