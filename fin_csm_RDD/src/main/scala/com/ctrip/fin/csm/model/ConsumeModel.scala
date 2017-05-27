package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.{CSMUtil, ModelUtil}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by zhongxl on 2016/10/13.
  * Updated by zyong on 2016/10/20.
  */
class ConsumeModel() extends SCMModel("consume") {

  /**
    * 打分
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

    val score_1 = CSMUtil.concatDataFrame(predict_1_temp, predict_2_temp)
    val score_2 = score_1.withColumn("score_RF", score_1("prediction_1") * 500 + 350)

    //-----------自此可以直接只返回RF------------------------------------------------------//
    val score_3 = score_2.withColumn("score_GBT", score_1("prediction_2") * 500 + 350)
    val score_df = score_3.withColumn("score_consuming", score_3("score_RF") * 0.5 + score_3("score_GBT") * 0.5).select("score_consuming")
    //-------- --按比例混合双模型的打分----------------------------------------------------//

    savePredictedData(sc, score_df)
    return score_df
  }

  override def processData(data: DataFrame, isUsedForTraining: Boolean): DataFrame = {

    //平均消费水平
    val data_1 = data.withColumn("ord_success_order_price",
      data("ord_success_order_amount") / data("ord_success_order_count"))

    //高星酒店消费
    val data_2 = data_1.withColumn("ord_success_first_class_order_price",
      data("ord_success_first_class_order_amount") / data("ord_success_first_class_order_count"))

    //海外酒店
    val data_3 = data_2.withColumn("ord_success_aboard_order_price",
      data("ord_success_aboard_order_amount") / data("ord_success_aboard_order_count"))

    //头等舱数据
    val data_4 = data_3.withColumn("ord_success_flt_first_class_order_price",
      data("ord_success_flt_first_class_order_amount") / data("ord_success_flt_first_class_order_count"))

    //机票海外订单
    val data_5 = data_4.withColumn("ord_success_flt_aboard_order_price",
      data("ord_success_flt_aboard_order_amount") / data("ord_success_flt_aboard_order_count"))

    //机票消费单价
    val data_6 = data_5.withColumn("ord_success_flt_order_price",
      data("ord_success_flt_order_amount") / data("ord_success_flt_order_count"))

    //高星酒店
    val data_7 = data_6.withColumn("ord_success_htl_first_class_order_price",
      data("ord_success_htl_first_class_order_amount") / data("ord_success_htl_first_class_order_count"))

    //海外酒店
    val data_8 = data_7.withColumn("ord_success_htl_aboard_order_price",
      data("ord_success_htl_aboard_order_amount") / data("ord_success_htl_aboard_order_count"))

    //酒店消费单价
    val data_9 = data_8.withColumn("ord_success_htl_order_price",
      data("ord_success_htl_order_amount") / data("ord_success_htl_order_count"))

    //火车票消费
    val data_10 = data_9.withColumn("ord_success_trn_order_price",
      data("ord_success_trn_order_amount") / data("ord_success_trn_order_count"))

    if (isUsedForTraining == true) {
      val data_new = data_10.na.fill(-1.0).select("uid_flag", "pro_advanced_date", "pro_htl_star_prefer", "pro_ctrip_profits",
        "ord_success_max_order_amount", "ord_success_avg_leadtime", "ord_cancel_order_count",
        "ord_success_order_type_count", "ord_success_order_acity_count", "ord_success_flt_last_order_days",
        "ord_success_flt_max_order_amount", "ord_success_flt_avg_order_pricerate",
        "ord_success_flt_order_acity_count", "ord_success_htl_last_order_days", "ord_success_htl_max_order_amount",
        "ord_success_htl_order_refund_ratio", "ord_success_htl_guarantee_order_count",
        "ord_success_htl_noshow_order_count", "ord_cancel_htl_order_count", "ord_success_trn_last_order_days",
        "ord_success_order_price", "ord_success_first_class_order_price", "ord_success_aboard_order_price",
        "ord_success_flt_first_class_order_price", "ord_success_flt_aboard_order_price",
        "ord_success_flt_order_price", "ord_success_htl_first_class_order_price", "ord_success_htl_aboard_order_price",
        "ord_success_htl_order_price", "ord_success_trn_order_price")
      return data_new
    }
    else {
      val data_new = data_10.na.fill(0).select("pro_advanced_date", "pro_htl_star_prefer", "pro_ctrip_profits",
        "ord_success_max_order_amount", "ord_success_avg_leadtime", "ord_cancel_order_count",
        "ord_success_order_type_count", "ord_success_order_acity_count", "ord_success_flt_last_order_days",
        "ord_success_flt_max_order_amount", "ord_success_flt_avg_order_pricerate",
        "ord_success_flt_order_acity_count", "ord_success_htl_last_order_days", "ord_success_htl_max_order_amount",
        "ord_success_htl_order_refund_ratio", "ord_success_htl_guarantee_order_count",
        "ord_success_htl_noshow_order_count", "ord_cancel_htl_order_count", "ord_success_trn_last_order_days",
        "ord_success_order_price", "ord_success_first_class_order_price", "ord_success_aboard_order_price",
        "ord_success_flt_first_class_order_price", "ord_success_flt_aboard_order_price",
        "ord_success_flt_order_price", "ord_success_htl_first_class_order_price", "ord_success_htl_aboard_order_price",
        "ord_success_htl_order_price", "ord_success_trn_order_price")
      return data_new
    }

  }

  def calculate(data: DataFrame, sqlContext: SQLContext, columnName: String) = {
    /** *
      * spark RDD 格式的数据转化为DataFrame格式时，需要声明一个隐式操作;
      * 声明隐式操作的生成DF方法目前只支持（INT LONG STRING）
      * */
    import sqlContext.implicits._
    //声明隐式操作
    val data_temp = data.map(col => (col.getDouble(0), col.getDouble(1)))
      .map(newcol => (newcol._2 / newcol._1))
      .map(i => i.toString)
      .toDF(columnName)

    // 将字符串的格式转回Double
    val data_temp2 = data_temp.select(data_temp(columnName).cast(DoubleType).as(columnName))

    //填补DF 中的null值为1，
    // 如果是许多列那么需要fill(1.0,seq("colname"))
    val data_new = data_temp2.na.fill(1.0)
    data_new
  }
}
