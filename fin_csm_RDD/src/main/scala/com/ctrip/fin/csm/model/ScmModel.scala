package com.ctrip.fin.csm.model

import java.io.{FileOutputStream, ObjectOutputStream, _}

import com.ctrip.fin.csm.utils.{CSMUtil, ModelUtil}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.mllib.linalg.Vector

import scala.reflect.io.Path

/**
  * Created by zyong on 2016/10/20.
  */
abstract class SCMModel(module: String) extends Serializable {

  val baseOutputPath: String = "/home/bifinread/csm"
  var scaler: StandardScalerModel = null
  val modulePath: String = s"${baseOutputPath}/${module}"
  val modelPath: String = s"${modulePath}/model"
  val predictedDataPath: String = s"${modulePath}/predict"
  val scalerPath: String = s"${modulePath}/scaler.obj"
  val savedDataFormat: String = "parquet"

  /**
    * 保存Scaler对象到本地目录
    *
    */
  def serializeScaler() = {
    //清理输出目录
    val outPath: Path = Path(modulePath)
    //try (outPath.deleteRecursively)
    outPath.createDirectory(true, true)

    val oos = new ObjectOutputStream(new FileOutputStream(scalerPath))
    oos.writeObject(scaler)
    oos.flush()
    oos.close()
  }

  /**
    * 反序列化读出数据
    *
    * @tparam T
    * @return
    */
  def deserializeScaler[T]() : StandardScalerModel= {
    val ois = new ObjectInputStream(new FileInputStream(scalerPath))
    ois.readObject().asInstanceOf[StandardScalerModel]
  }

  /**
    * 数据预处理
    * 通过利用DF 的特征来构建新的特征，通过添加trigger来区分是对训练数据的处理还是对测试数据的处理
    * @param data
    * @param isUsedForTraining
    */
  def processData(data: DataFrame, isUsedForTraining: Boolean = true): DataFrame

  /**
    * 训练模型保存训练模型结果与Scaler结果
    *
    * @param data
    * @param labelColumn
    */
  def train(data: DataFrame, labelColumn: String = "uid_flag") = {
    val sqlContext = data.sqlContext
    val sc = sqlContext.sparkContext

    val processedDF = processData(data, isUsedForTraining = true)
    val trainingLP = CSMUtil.dataFrame2LabelPointRDD(processedDF, labelColumn)
    // 归一化训练数据
    val vectors = trainingLP.map(p => p.features)
    scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    serializeScaler()
    val standardData = trainingLP.map(point =>
      LabeledPoint(point.label, scaler.transform(point.features))
    )
    ModelUtil.trainModel(standardData, modelPath)
  }

  def getGBTModel(sc: SparkContext): GradientBoostedTreesModel ={
    GradientBoostedTreesModel.load(sc, s"${modelPath}/GBT")
  }

  def getRFModel(sc: SparkContext): RandomForestModel = {
    RandomForestModel.load(sc, s"${modelPath}/RF")
  }

  /**
    * 预测数据标准化
    *
    * @param data
    * @param isUsedForTraining 区别数据是用于训练还是测试
    * @return
    */
  def standardTestData(data: DataFrame, isUsedForTraining: Boolean = false): RDD[Vector] = {
    val processedData = processData(data, isUsedForTraining)
    // 转换数据为Vector
    val lpData = CSMUtil.dataFrame2VectorRDD(processedData)
    // 归一化测试数据
    deserializeScaler().transform(lpData)
  }

  /**
    * 存储打过分的数据
    * @param sc
    * @param data
    */
  def savePredictedData(sc: SparkContext, data: DataFrame) = {
    CSMUtil.deleteHdfsPath(sc, predictedDataPath)
    data.write.format(savedDataFormat).save(predictedDataPath)
  }

  /**
    * 读取打过分的数据
    * @param sqlContext
    * @return
    */
  def readPredictedData(sqlContext: SQLContext): DataFrame = {
    sqlContext.read.format(savedDataFormat).load(predictedDataPath)
  }
}
