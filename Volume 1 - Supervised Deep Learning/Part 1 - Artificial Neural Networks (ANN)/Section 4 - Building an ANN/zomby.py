
###############################################################################
#
# N O R M A L I Z A T I O N
###############################################################################
#
#X_scaler, Y_scaler = ImageDataProvider(path_to_dataset + '/' + '[tv][ra][al]*/*.tiff',
#                                     data_suffix=img_type,
#                                     shuffle_data=False,
#                                     mask_suffix='_mask' + img_type).load_data()
#print(np.min(X_scaler), np.max(X_scaler), np.mean(X_scaler), np.std(X_scaler))
#
##Feature Scaling
#from sklearn.preprocessing import StandardScaler
#
#sc = StandardScaler()
#X_scaler = sc.fit_transform(X_scaler.reshape(-1, 1)).reshape(X_scaler.shape)
#print(sc.mean_, sc.scale_)
#
#
#from sklearn.externals import joblib
#joblib.dump(sc, "scaler.save") 
#print(np.mean(X_scaler), np.std(X_scaler))

#Feature Scaling
#from sklearn.externals import joblib
#sc = joblib.load("scaler.save")
#print(sc.mean_, sc.scale_)

#from sklearn.preprocessing import StandardScaler




# Provide the same seed and keyword arguments to the fit and flow methods
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)




#from sklearn.utils import class_weight
#class_weight = class_weight.compute_class_weight('balanced', [0., 1.], y_train.ravel())
#print(class_weight)



    
##############################################################################
# CURVA ROC y Precision-Recall curve
############################################################################### 

# Curvas ROC
#_, _, _ = roc(y_val.ravel(), y_pred.ravel())

# Compute the average precision score
#from sklearn.metrics import average_precision_score
#average_precision = average_precision_score(y_val.ravel(), y_pred.ravel())
#
#print('Average precision-recall score: {0:0.2f}'.format(average_precision))
#
#
#from sklearn.metrics import precision_recall_curve
#import matplotlib.pyplot as plt
#
#precision, recall, thresholds = precision_recall_curve(y_val.ravel(), y_pred.ravel())
#
#plt.step(recall, precision, color='b', alpha=0.2, where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
## Plot the Precision-Recall curve
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#plt.show()



###############################################################################
# X V A L
###############################################################################    
#import glob
#import matplotlib.pyplot as plt
#
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
#from sklearn.metrics import jaccard_similarity_score
#
#for history, params, _ in grid_search:  
#    
#    y_pred = history.model.predict(X_val, batch_size=16, verbose=0)
#    y_pred[y_pred>0.5] = 1.0; y_pred[y_pred<=0.5] = 0.0
##        confusion_matrixes[i] = confusion_matrix(y_test.ravel(), y_pred.ravel(),
##                          labels=[0., 1.])
##        _, recalls[i], _, _ = precision_recall_fscore_support(y_test.ravel(), y_pred.ravel())
#    
#    # print(jaccard_similarity_score(y_test.ravel(), y_pred.ravel()))
#    f1_pred = [np_dsc(y_true, y) for y_true, y in zip(y_val, y_pred)]
#    
#    print(params)
#    plt.plot(f1_pred, color='green')
#    plt.show()
#    i = i +1;
#    
#    print('-'*30)
#    # print('cm=', confusion_matrixes[i])
#    # print('(precision, recall)=', recalls[i][0], recalls[i][1])
#   
#    for n in np.argsort(np.array(f1_pred))[0:2]:
#        #n = np.random.randint(X_test.shape[0])
#        print("number of frame", n, f1_pred[n])
#        fig, ax = plt.subplots(1, 3, figsize=(12, 18))
#        ax[0].imshow(X_val[n,:,:,0], cmap='gray')
#        ax[0].set_title('CT')
#        ax[1].imshow(y_val[n,:,:,0], cmap='gray')
#        ax[1].set_title('Ground Truth')
#        ax[2].imshow(y_pred[n,:,:,0], cmap='gray')
#        ax[2].set_title('U-Net')
#        plt.show()
#
#    for n in np.argsort(-np.array(f1_pred))[0:2]:
#        #n = np.random.randint(X_test.shape[0])
#        print("number of frame", n, f1_pred[n])
#        fig, ax = plt.subplots(1, 3, figsize=(12, 18))
#        ax[0].imshow(X_val[n,:,:,0], cmap='gray')
#        ax[0].set_title('CT')
#        ax[1].imshow(y_val[n,:,:,0], cmap='gray')
#        ax[1].set_title('Ground Truth')
#        ax[2].imshow(y_pred[n,:,:,0], cmap='gray')
#        ax[2].set_title('U-Net')
#        plt.show()