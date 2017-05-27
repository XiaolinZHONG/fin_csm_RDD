package com.ctrip.fin.csm.utils

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.rdd.RDD

/**
  * Created by zhongxl on 2016/10/13.
  */
object ModelUtil extends Serializable {

  /**
    * 使用MLLIB中的RF 和 GBT 回归分析
    *
    * @param data
    * @param modelPath
    */
  def trainModel(data: RDD[LabeledPoint], modelPath: String) = {

    val rfModelPath: String = s"${modelPath}/RF"
    val gbtModelPath: String = s"${modelPath}/GBT"

    val sc = data.context

    CSMUtil.deleteHdfsPath(sc, modelPath)

    val splits = data.randomSplit(Array(0.75, 0.25))
    val (trainingData, testData) = (splits(0), splits(1))

    // 使用RandomForest
    val numClasses = 2 //二分类
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 30 // 树的数目
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 10
    val maxBins = 32

    trainingData.cache()
    val modelRfr = RandomForest.trainRegressor(
      trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins
    )

    // 交叉验证集
    val labelsAndPredictions_rfr = testData.map { point =>
      val prediction = modelRfr.predict(point.features)
      (point.label, prediction)
    }
    evaluateReport(labelsAndPredictions_rfr) //评估模型
    evaluate_report_classification(labelsAndPredictions_rfr, testData)

    //模型的保存与重载
    modelRfr.save(sc, rfModelPath)

    // 使用GBT regression
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 50 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 10
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model_gbt = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // 交叉验证集
    val labelsAndPredictions_gbt = testData.map { point =>
      val prediction = model_gbt.predict(point.features)
      (point.label, prediction)
    }
    //evaluate_report(labelsAndPredictions_gbt)//评估模型
    evaluate_report_classification(labelsAndPredictions_gbt, testData)

    //模型的保存与重载
    model_gbt.save(sc, gbtModelPath)

  }

  /**
    * 评估分类模型的结果
    *
    * @param labelAndPreds
    * @param testData
    */
  def evaluate_report_classification(labelAndPreds: RDD[(Double, Double)], testData: RDD[LabeledPoint]): Unit = {

    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val score = metrics.scoreAndLabels
    /*score.foreach { case (t, p) =>
      println(s"label: $t,prediction: $p")
    }*/
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println(s"Area under ROC = ${auROC}")

  }

  /**
    * 评估回归模型的结果
    *
    * @param labelAndPredict
    */
  def evaluateReport(labelAndPredict: RDD[(Double, Double)]): Unit = {
    /*val score = labelAndPredict.foreach { case (v, p) =>
      println(s"label: $v,prediction: $p")
    }*/
    val testMSE = labelAndPredict.map { case (v, p) => math.pow((v - p), 2) }.mean() //使用case 时 中括号
    println("Test Mean Squared Error = " + testMSE)
  }


}
