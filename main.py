x_train = data.x_train
x_val = data.x_val
y_train = data.y_train
y_val = data.y_val
x_test = data.x_test
y_test = data.y_test
train_samples = len(x_train)
val_samples = len(x_val)
test_samples = len(x_test)
test_samples

criterion = nn.BCELoss()
print_every = 1
print_loss_total = 0.0
train_loss = 0.0
max_acc = 0.7
par = 0.5

model_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
model_trainable_parameters = model_trainable_parameters + final_par
model_trainable_parameters = tuple(model_trainable_parameters)
model_optimizer = optim.Adam(model_trainable_parameters, lr=learning_rate)

#from helper import Helper
help_fn = Helper()

#run to load the base model

#model.load_state_dict(torch.load("../model_weights/model_weights.pt"))
model.eval()
model.train()


start = time.time()
print('Beginning Model Training.\n')
batch_size = 16

for epoch in range(0, num_iters):
    model_loss1 = 0.0
    gen_loss1 = 0.0
    dis_loss1 = 0.0
    fin_loss1 = 0.0
    train_loss1 = 0.0
    model_loss2 = 0.0
    gen_loss2 = 0.0
    dis_loss2 = 0.0
    fin_loss2 = 0.0
    train_loss2 = 0.0
    val_loss = 0.0
    for i in range(0, train_samples, batch_size):
        input_variables = x_train[i:i+batch_size]
        similarity_scores = y_train[i:i+batch_size]
        
        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        # Make a tensor for the similarity scores
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if similarity_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]

        model_optimizer.zero_grad()
        loss_s = 0.0
        
        optimizerG.zero_grad()
        loss_g= 0.0
        
        optimizerD.zero_grad()
        loss_d= 0.0

        loss_f = 0.0

        # Initialise the hidden state and pass through the maLSTM
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, similarity_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda()
        
        
        # Generator
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        #print(discrimm_classes)
        
        # Update generator loss
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
        
        loss_f += criterion(final_labels, sim_scores_2d)
        
        com_loss = (0.6*loss_g) + loss_f + loss_d
        
        com_loss.backward()
        
        model_optimizer.step()
        optimizerG.step()
        
        
        fin_loss1 += loss_f
        model_loss1 += loss_s
        gen_loss1 += loss_g
        dis_loss1 += loss_d
    
        train_loss1 += com_loss
        
        
    for i in range(0, train_samples, batch_size):
        input_variables = x_train[i:i+batch_size]
        similarity_scores = y_train[i:i+batch_size]
        
        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        # Make a tensor for the similarity scores
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if similarity_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]
        
        optimizerD.zero_grad()
        loss_d = 0.0
        
        model_optimizer.zero_grad()
        loss_s = 0.0
        
        optimizerG.zero_grad()
        loss_g= 0.0
        
        loss_f = 0.0

        # Initialise the hidden state and pass through the maLSTM
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, similarity_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda()
        
        
        # Generator
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
      
        loss_f += criterion(final_labels, sim_scores_2d)
        com_loss = (0.6*loss_g) + loss_f + loss_d
        
        com_loss.backward()
        
        optimizerD.step()
        
        
        fin_loss2 += loss_f
        model_loss2 += loss_s
        gen_loss2 += loss_g
        dis_loss2 += loss_d
    
        train_loss2 += com_loss
    
    
    a_scores = []
    p_scores = []
    corr = 0
    fin_lossv = 0.0
    model_lossv = 0.0
    gen_lossv = 0.0
    dis_lossv = 0.0
    for i in range(0, test_samples, batch_size):
        input_variables = x_test[i:i+batch_size]
        actual_scores = y_test[i:i+batch_size]

        sequences_1 = [sequence[0] for sequence in input_variables]
        sequences_2 = [sequence[1] for sequence in input_variables]
        batch_size = len(sequences_1)
        
        sim_scores_2d = torch.zeros([batch_size, 2])
        for j in range(batch_size):
          if actual_scores[j] == 0:
            sim_scores_2d[j] = fake_label
          else:
            sim_scores_2d[j] = real_label
            
        sim_scores_2d = sim_scores_2d.cuda()

        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]

        loss = 0.0
        loss_d = 0.0
        loss_g = 0.0
        loss_f = 0.0
        loss_s = 0.0
        
        hidden = model.init_hidden(batch_size)
        output_scores, ehs1, ehs2 = model([sequences_1, sequences_2], hidden)
        
        output_scores = output_scores.view(-1)
        
        loss_s += criterion(output_scores, actual_scores)
        
        ehs1 = ehs1.cuda()
        ehs2 = ehs2.cuda() 
        gen_feature = netG(ehs2)
        
        # 1. Discriminator for the real class
        discrimm_classes = netD(ehs1)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = real_label
          
        labels = labels.cuda()
        
        loss_d += criterion(discrimm_classes, labels)
        
        
        # 2. Discriminator for the fake class
        discrimm_classes = netD(gen_feature)
        labels = torch.zeros(batch_size, 2)
        for j in range(batch_size):
          labels[j] = fake_label
          
        labels = labels.cuda()
          
        loss_d += criterion(discrimm_classes, labels)
        
        loss_g += criterion(discrimm_classes, sim_scores_2d)
        
        d_feature = dropout_layer(gen_feature)
        
        cat_feature = torch.zeros([batch_size, len(d_feature[0])+1])
        for j in range(batch_size):
          for k in range(100):
            cat_feature[j][k] = d_feature[j][k]
          cat_feature[j][100] = output_scores[j]
          
        cat_feature = cat_feature.cuda()
        
        final_labels = net_final(cat_feature)
        
        loss_f += criterion(final_labels, sim_scores_2d)
        
        loss = loss_f + loss_d + (0.6*loss_g)
        
        fin_lossv += loss_f
        model_lossv += loss_s
        gen_lossv += loss_g
        dis_lossv += loss_d
        
        val_loss += loss
        
        for j in range(0, batch_size):
          acts = actual_scores[j].data.cpu().numpy()
          preds = final_labels[j].data.cpu().numpy()
          a_scores.append(acts)

          if preds[0] >= 0.5 and acts == 0:
            corr = corr+1
            p_scores.append(0)
          elif preds[1] >= 0.5 and acts == 1:
            corr = corr+1
            p_scores.append(1)
          elif preds[0] >=0.5:
            p_scores.append(0)
          else:
            p_scores.append(1)
          
    
    if epoch % print_every == 0:
        print('%s (%d)' % (help_fn.time_slice(start, (epoch+1) / num_iters), epoch))
        print("Train Loss    " + str(train_loss2.data.cpu().numpy()) + "    Val loss    " + str(val_loss.data.cpu().numpy()))
        print("LSTM loss 1   " + str(model_loss1.data.cpu().numpy()) + "    Gen loss 1   " + str(gen_loss1.data.cpu().numpy()) + "    Dis loss 1   " + str(dis_loss1.data.cpu().numpy()) + "    Fin loss 1   " + str(fin_loss1.data.cpu().numpy()))
        print("LSTM loss 2   " + str(model_loss2.data.cpu().numpy()) + "    Gen loss 2   " + str(gen_loss2.data.cpu().numpy()) + "    Dis loss 2   " + str(dis_loss2.data.cpu().numpy()) + "    Fin loss 2   " + str(fin_loss2.data.cpu().numpy()))
        print("LSTM loss v   " + str(model_lossv.data.cpu().numpy()) + "    Gen loss v   " + str(gen_lossv.data.cpu().numpy()) + "    Dis loss v   " + str(dis_lossv.data.cpu().numpy()) + "    Fin loss v   " + str(fin_lossv.data.cpu().numpy()))
        print(" Test Accuracy    " + str(corr/len(a_scores)) + "    f1 score    " + str(f1_score(p_scores, a_scores)))
        
        acc = corr/len(a_scores)
        
        if acc > max_acc :
          max_acc = acc
          torch.save(model.state_dict(), "../model_weights/model_weights.pt")
          torch.save(netG.state_dict(), "../model_weights/netG_weights.pt" )
          torch.save(netD.state_dict(), "../model_weights/netD_weights.pt")
          torch.save(net_final.state_dict(), "../model_weights/netfinal_weights.pt")
          print("Model Saved!")
