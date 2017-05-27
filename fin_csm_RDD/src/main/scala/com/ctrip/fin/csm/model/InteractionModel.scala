package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.ModelUtil
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by zhongxl on 2016/10/13.
  * Updated by zyong on 2016/10/20.
  */
class InteractionModel extends SCMModel("interaction") {

  /**
    * 打分
    *
    * @return
    */
  def predict(data: DataFrame): DataFrame = {

    val sqlContext = data.sqlContext
    val sc = sqlContext.sparkContext

    val standardData = standardTestData(data)
    val prediction_GBT = getGBTModel(sc).predict(standardData)
    val prediction_RF = getRFModel(sc).predict(standardData)

    //融合预测的数据
    import sqlContext.implicits._
    //声明隐式操作
    val predict_1 = prediction_RF.map(i => i.toString).toDF("prediction_1")
    val predict_2 = prediction_GBT.map(i => i.toString).toDF("prediction_2")

    val predict_1_temp = predict_1.select(predict_1("prediction_1").cast(DoubleType).as("prediction_1"))
    val predict_2_temp = predict_2.select(predict_2("prediction_2").cast(DoubleType).as("prediction_2"))

    val score_1 = predict_1_temp.withColumn("score_interaction", predict_1_temp("prediction_1") * 500 + 350).select("score_interaction")
    //-------------------------------------------自此可以直接只返回RF----------------------------------------------------//
    //    val score_2        = score_1.withColumn("score_GBT",score_1("prediction_2")*500+350)
    //    val score_df       = score_2.withColumn("score",score_2("score_RF")*0.6+score_2("score_GBT")*0.4).select("score")
    //    score_df
    savePredictedData(sc, score_1)
    return score_1
  }

  /**
    * 数据预处理
    * 通过利用DF 的特征来构建新的特征，通过添加isUsedForTraining来区分是对训练数据的处理还是对测试数据的处理
    *
    * @param data
    * @param isUsedForTraining
    */
  override def processData(data: DataFrame, isUsedForTraining: Boolean): DataFrame = {
    var dataNew: DataFrame = null
    if (isUsedForTraining == true){
      dataNew = data.select("uid_flag", "voi_complaint_count", "voi_complrefund_count",
        "voi_comment_count", "acc_loginday_count", "pro_validpoints",
        "pro_base_active", "pro_ctrip_profits", "pro_customervalue")
    } else {
      dataNew = data.select("voi_complaint_count", "voi_complrefund_count",
        "voi_comment_count", "acc_loginday_count", "pro_validpoints",
        "pro_base_active", "pro_ctrip_profits", "pro_customervalue")
    }
    dataNew
  }
}
