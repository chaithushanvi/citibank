
      def prediction():
          if request.method=="POST":
              gfg=request.form['gfg']
              ghfc=request.form['ghfc']
              ghpsc=request.form['ghpsc']
              ghwr=request.form['ghwr']
              gor=request.form['gor']
              gw=request.form['gw']
              tlf=request.form['tlf']
              tla=request.form['tla']
              time=request.form['time']
              values=[[float(gfg),float(ghfc),float(ghpsc),float(ghwr),float(gor),float(gw),float(tlf),float(tla),float(time))]]
              xgbr=xgb.XGBRegressor(learning_rate=0.4,n_estimators=200)
              model=xgbr.fit(x_train,y_train)
              df_pred=pd.Dataframe(values[0],index=x_test.columns).transpose()
              pred=model.predict(df_pred)
              return render_template('prediction.html',msg='success',result=pred)
           return render_template('prediction.html')



CODE FOR RANDOM FOREST:
   
      rfr=RandomFprestRegressor(n_estimators=50,max_depth=14)
      model1=rfr.fit(x_train,y_train)
      pred1=model1.predict(x_test)
      score=r2_score(y_test,pred1)
      return render_template('model.html',msg='accuracy',result=round(score,4),selected='RANDOM FOREST REGRESSOR')


CODE FOR SUPPORT VECTOR MACHINE:
 

      svr=SVR(kernel='rbf')
      x_train=x_train[:15000]
      y_train=y_train[:15000]
      model3=svr.fit(x_train[:15000],y_train[:15000])
      pred3=model3.predict(x_test[:15000])
      score=r2_score(y_test[:15000],pred3)
      return render_template('model.html',msg='accuracy',result=round(score,4),selected='SUPPORT VECTOR REGRESSOR')


CODE FOR XGBOOST ALGORITHM:

    xgbr=xgb.XGBRegressor(learning_rate=0.4,n_estimators=200)
    model2=xgbr.fit(x_train,y_train)
    pred2=model2.predict(x_test)
    score=r2_score(y_test,pred2)
    return render_template('model.html',msg='accuracy',result=round(score,4),selected='XGBOOST REGRESSOR')
