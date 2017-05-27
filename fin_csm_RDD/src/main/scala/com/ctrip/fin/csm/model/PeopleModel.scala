package com.ctrip.fin.csm.model

import com.ctrip.fin.csm.utils.{CSMUtil, ModelUtil}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType

/**
  * Created by zhongxl on 2016/10/13.
  * Updated by zyong on 2016/10/20.
  */
class PeopleModel extends SCMModel("people") {

  /**
    * 打分
    *
    * @return
    */
  def predict(data: DataFrame): DataFrame = {

    val sqlContext = data.sqlContext
    val sc = sqlContext.sparkContext
    /** *
      * 测试数据处理
      * */

    val standardData = standardTestData(data)
    val prediction_GBT = getGBTModel(sc).predict(standardData)
    val prediction_RF = getRFModel(sc).predict(standardData)

    //    //融合预测的数据
    import sqlContext.implicits._
    //声明隐式操作
    val predict_1 = prediction_RF.map(i => i.toString).toDF("prediction_1")
    val predict_2 = prediction_GBT.map(i => i.toString).toDF("prediction_2")

    //    val score_1= prediction_GBT.zip(prediction_RF)
    val predict_1_temp = predict_1.select(predict_1("prediction_1").cast(DoubleType).as("prediction_1"))
    val predict_2_temp = predict_2.select(predict_2("prediction_2").cast(DoubleType).as("prediction_2"))
    val score_1 = CSMUtil.concatDataFrame(predict_1_temp, predict_2_temp)
    val score_2 = score_1.withColumn("score_RF", score_1("prediction_1") * 500 + 350)

    //-----------自此可以直接只返回RF------------------------------------------------------//
    val score_3 = score_2.withColumn("score_GBT", score_1("prediction_2") * 500 + 350)
    val score_df = score_3.withColumn("score_people", score_3("score_RF") * 0.5 + score_3("score_GBT") * 0.5).select("score_people")
    //-------- --按比例混合双模型的打分----------------------------------------------------//

    savePredictedData(sc, score_df)
    score_df
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
    if (isUsedForTraining == true) {
      dataNew = data.select("uid_flag", "uid_grade", "uid_dealorders", "uid_emailvalid", "uid_age", "uid_mobilevalid",
        "uid_addressvalid", "uid_isindentify", "uid_authenticated_days", "uid_signupdays",
        "uid_signmonths", "uid_lastlogindays", "uid_samemobile", "ord_success_order_cmobile_count",
        "com_mobile_count", "pro_generous_stingy_tag", "pro_base_active",
        "pro_customervalue", "pro_phone_type", "pro_validpoints", "pro_htl_consuming_capacity")
    } else {
      dataNew = data.select("uid_grade", "uid_dealorders", "uid_emailvalid", "uid_age", "uid_mobilevalid",
        "uid_addressvalid", "uid_isindentify", "uid_authenticated_days", "uid_signupdays",
        "uid_signmonths", "uid_lastlogindays", "uid_samemobile", "ord_success_order_cmobile_count",
        "com_mobile_count", "pro_generous_stingy_tag", "pro_base_active",
        "pro_customervalue", "pro_phone_type", "pro_validpoints", "pro_htl_consuming_capacity")
    }
    dataNew
  }
}
