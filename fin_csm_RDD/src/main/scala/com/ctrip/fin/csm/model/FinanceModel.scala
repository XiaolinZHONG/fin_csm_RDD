package com.ctrip.fin.csm.model

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType

/**
  * Created by zhongxl on 2016/10/13.
  * Updated by zyong on 2016/10/20.
  */
class FinanceModel extends SCMModel("finance") {

  /**
    * 打分
    *
    * @param data
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

    val score_1 = predict_1_temp.withColumn("score_fanacial", predict_1_temp("prediction_1") * 500 + 350).select("score_fanacial")

    //-------------------------------------------自此可以直接只返回RF----------------------------------------------------//
    //    val score_2        = score_1.withColumn("score_GBT",score_1("prediction_2")*500+350)
    //    val score_df       = score_2.withColumn("score",score_2("score_RF")*0.6+score_2("score_GBT")*0.4).select("score")
    //    score_df
    savePredictedData(sc, score_1)
    return score_1
  }

  override def processData(data: DataFrame, isUsedForTraining: Boolean): DataFrame = {

    val data_1 = data.withColumn("cap_balance",
      data("cap_tmoney_balance") + data("cap_wallet_balance") + data("cap_wallet_balance"))

    val data_2 = data_1.withColumn("bil_pays_ratio",
      data("bil_paysord_count") / data("bil_payord_count"))

    val data_3 = data_2.withColumn("bil_pays_credit_ratio",
      data("bil_paysord_credit_count") / data("bil_payord_credit_count"))

    val data_4 = data_3.withColumn("bil_pays_debit_ratio",
      data("bil_paysord_debit_count") / data("bil_payord_debit_count"))

    val data_5 = data_4.withColumn("ord_success_first_class_order_price",
      data("ord_success_first_class_order_amount") / data("ord_success_first_class_order_count"))

    val data_6 = data_5.withColumn("ord_success_htl_aboard_order_price",
      data("ord_success_htl_aboard_order_amount") / data("ord_success_htl_aboard_order_count"))

    if (isUsedForTraining == true) {
      val data_new = data_6.na.fill(0)
        .select("uid_flag", "voi_complrefund_count", "fai_lackbalance", "bil_refundord_count",
          "bil_ordertype_count", "bil_platform_count", "pro_htl_star_prefer",
          "pro_htl_consuming_capacity", "pro_phone_type", "ord_success_max_order_amount",
          "ord_total_order_amount", "ord_success_flt_first_class_order_count",
          "ord_success_trn_max_order_amount", "ord_success_htl_first_class_order_count",
          "ord_success_htl_max_order_amount", "ord_success_aboard_order_count",
          "cap_balance", "ord_success_htl_aboard_order_price", "ord_success_first_class_order_price",
          "bil_pays_debit_ratio", "bil_pays_credit_ratio", "bil_pays_ratio")
      return data_new
    }
    else {
      val data_new = data_6.na.fill(0).select("voi_complrefund_count", "fai_lackbalance", "bil_refundord_count",
        "bil_ordertype_count", "bil_platform_count", "pro_htl_star_prefer",
        "pro_htl_consuming_capacity", "pro_phone_type", "ord_success_max_order_amount",
        "ord_total_order_amount", "ord_success_flt_first_class_order_count",
        "ord_success_trn_max_order_amount", "ord_success_htl_first_class_order_count",
        "ord_success_htl_max_order_amount", "ord_success_aboard_order_count",
        "cap_balance", "ord_success_htl_aboard_order_price", "ord_success_first_class_order_price",
        "bil_pays_debit_ratio", "bil_pays_credit_ratio", "bil_pays_ratio")
      return data_new
    }
  }
}
