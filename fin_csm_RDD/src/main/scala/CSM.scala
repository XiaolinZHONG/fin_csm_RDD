package com.ctrip.fin.csm


import com.ctrip.fin.csm.model.{FinanceModel, _}
import com.ctrip.fin.csm.utils.{CSMUtil, ModelUtil}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by zyong on 2016/10/13.
  */
object CSM {

  var sc: SparkContext = null
  var sqlContext: SQLContext = null
  val baseOutputPath = "/user/bifinread/csm/"

  /**
    * 训练模型
    * 保存模型与Scaler对象
    *
    * @param moduleName
    * @return
    */
  def train(moduleName: String, dataPath: String) = {
    val data = CSMUtil.readCSV(dataPath, sc, sqlContext)
    moduleName match {
      case "consume" => new ConsumeModel().train(data)
      case "finance" => new FinanceModel().train(data)
      case "interaction" => new InteractionModel().train(data)
      case "people" => new PeopleModel().train(data)
      case "relation" => new RelationModel().train(data)
      case _ =>
    }
  }

  /**
    * 预测结果
    *
    * @param moduleName
    * @param dataPath
    * @return
    */
  def predict(moduleName: String, dataPath: String) = {
    val data = CSMUtil.readTest(dataPath, sc, sqlContext)
    val predictDF: DataFrame = moduleName match {
      case "consume" => new ConsumeModel().predict(data)
      case "finance" => new FinanceModel().predict(data)
      case "interaction" => new InteractionModel().predict(data)
      case "people" => new PeopleModel().predict(data)
      case "relation" => new RelationModel().predict(data)
    }
    predictDF.show(5)
  }

  /**
    * 合并结果集
    */
  def combineResult() = {
    val consumeDF = new ConsumeModel().readPredictedData(sqlContext)
    val financeDF = new FinanceModel().readPredictedData(sqlContext)
    val interactionDF = new InteractionModel().readPredictedData(sqlContext)
    val peopleDF = new PeopleModel().readPredictedData(sqlContext)
    val relationDF = new RelationModel().readPredictedData(sqlContext)
    val combinedDF = CSMUtil.concatDataFrame(consumeDF, financeDF, interactionDF, peopleDF, relationDF)

    val score = combinedDF.withColumn("score_all",
      combinedDF("score_people") * 0.3 +
        combinedDF("score_consuming") * 0.25 +
        combinedDF("score_fanacial") * 0.2 +
        combinedDF("score_interaction") * 0.15 +
        combinedDF("score_relation") * 0.1)

    score.show(10)

    //val score_with_label = CSMUtil.concatDataFrame(score.select("score_all"), data_tst.select("score"), sqlContext)
  }

  def main(args: Array[String]): Unit = {

    val appName = args(0)
    val appType = args(1) // train or predict
    val moduleName = args(2) // sub-model name
    val dataPath = args(3)

    // 声明
    val conf = new SparkConf().setAppName(appName).setMaster("local")
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)

    appType match {
      case "train" => train(moduleName, dataPath)
      case "predict" => predict(moduleName, dataPath)
      case "combine" => combineResult()
    }

  }
}
