package com.ctrip.fin.csm.utils

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * Created by zyong on 2016/10/20.
  */
object CSMUtil extends Serializable {

  def readCSV(address: String, sc: SparkContext, sqlContext: SQLContext): DataFrame = {

    val rawData = sc.textFile(address)
    val schemaString = "uid_grade,uid_age,uid_lastlogindays,uid_dealorders,uid_emailvalid," +
      "uid_mobilevalid,uid_addressvalid,uid_isindentify,uid_authenticated_days,uid_samemobile," +
      "uid_signupdays,uid_signmonths,uid_haspaypwd,pro_advanced_date,pro_generous_stingy_tag," +
      "pro_htl_star_prefer,pro_htl_consuming_capacity,pro_phone_type,pro_validpoints," +
      "pro_base_active,pro_secretary,pro_agent,pro_ctrip_profits,pro_customervalue," +
      "pro_generousindex_fltn,pro_lastyearprofits_fltn,pro_pricesensitivity_fltn,pro_ismarketing," +
      "ord_success_last_order_days,ord_success_max_order_amount,ord_success_order_count," +
      "ord_success_order_quantity,ord_success_order_amount,ord_success_avg_leadtime," +
      "ord_total_order_count,ord_total_order_quantity,ord_total_order_amount,ord_cancel_order_count," +
      "ord_refund_order_count,ord_success_first_class_order_count,ord_success_first_class_order_amount," +
      "ord_success_order_type_count,ord_success_self_order_count,ord_success_self_order_amount," +
      "ord_success_aboard_order_count,ord_success_aboard_order_amount,ord_success_order_acity_count," +
      "ord_success_order_cmobile_count,ord_success_flt_last_order_days,ord_success_flt_first_class_order_count," +
      "ord_success_flt_first_class_order_amount,ord_success_flt_max_order_amount,ord_success_flt_avg_order_pricerate," +
      "ord_success_flt_aboard_order_count,ord_success_flt_aboard_order_amount,ord_success_flt_order_count," +
      "ord_success_flt_order_amount,ord_success_flt_order_acity_count,ord_cancel_flt_order_count," +
      "ord_refund_flt_order_count,ord_success_htl_last_order_days,ord_success_htl_first_class_order_count," +
      "ord_success_htl_first_class_order_amount,ord_success_htl_max_order_amount,ord_success_htl_order_refund_ratio," +
      "ord_success_htl_aboard_order_count,ord_success_htl_aboard_order_amount,ord_success_htl_guarantee_order_count," +
      "ord_success_htl_noshow_order_count,ord_success_htl_avg_order_pricerate,ord_success_htl_order_count," +
      "ord_success_htl_order_amount,ord_cancel_htl_order_count,ord_refund_htl_order_count,ord_success_pkg_last_order_days," +
      "ord_success_pkg_first_class_order_count,ord_success_pkg_first_class_order_amount,ord_success_pkg_max_order_amount," +
      "ord_success_pkg_aboard_order_count,ord_success_pkg_aboard_order_amount,ord_success_pkg_order_count," +
      "ord_success_pkg_order_amount,ord_cancel_pkg_order_count,ord_refund_pkg_order_count,ord_success_trn_last_order_days," +
      "ord_success_trn_first_class_order_count,ord_success_trn_first_class_order_amount,ord_success_trn_max_order_amount," +
      "ord_success_trn_order_count,ord_success_trn_order_amount,ord_cancel_trn_order_count,ord_refund_trn_order_count," +
      "com_passenger_count,com_idno_count,com_mobile_count,com_has_child,fai_lackbalance,fai_risk,fai_putwrong," +
      "fai_invalid,bil_payord_count,bil_paysord_count,bil_refundord_count,bil_payord_credit_count," +
      "bil_payord_debit_count,bil_paysord_credit_count,bil_paysord_debit_count,bil_ordertype_count," +
      "bil_platform_count,cap_tmoney_balance,cap_wallet_balance,cap_refund_balance,cap_total_balance," +
      "cap_withdrow_count,cap_withdrow_amount,acc_loginday_count,acc_pwd_count,acc_paypwd_count," +
      "acc_bindmobile_count,voi_complaint_count,voi_complrefund_count,voi_comment_count,uid_flag"

    val fields = schemaString.split(",").map(fieldName => StructField(fieldName, DoubleType, true))
    val schema = StructType(fields)
    val noheader = rawData.filter(line => !line.contains("uid_grade"))
    val myfile2 = noheader.map(_.split(",")).map(p => Row.fromSeq(p.map(x => x.toDouble).toSeq))
    val df = sqlContext.createDataFrame(myfile2, schema)
    val end_Time = System.currentTimeMillis() //获取

    df.show(1)
    df.printSchema()
    return df
  }

  /**
    * 读取csv,txt文件
    * 同时数据的columns name 必须单独写成下面的格式；
    * 不可以在中间随意增加columns；
    *
    * @param address 返回SPARK DATA_FRAME
    * @param sc
    * @param sqlContext
    * @return 返回值为spark dataframe，但是可能会包含原来文件的header。
    */
  def readTest(address: String, sc: SparkContext, sqlContext: SQLContext): DataFrame = {

    val start_Time = System.currentTimeMillis() //获取
    val myfile = sc.textFile(address)
    val schemaString = "id,uid_grade,uid_age,uid_lastlogindays,uid_dealorders,uid_emailvalid,uid_mobilevalid," +
      "uid_addressvalid,uid_isindentify,uid_authenticated_days,uid_signupdays,uid_signmonths,pro_advanced_date," +
      "pro_generous_stingy_tag,pro_htl_star_prefer,pro_htl_consuming_capacity,pro_phone_type,pro_validpoints," +
      "pro_base_active,pro_secretary,pro_agent,pro_ctrip_profits,pro_customervalue,pro_generousindex_fltn," +
      "pro_lastyearprofits_fltn,pro_pricesensitivity_fltn,ord_success_last_order_days,ord_success_max_order_amount," +
      "ord_success_order_count,ord_success_order_quantity,ord_success_order_amount,ord_success_avg_leadtime," +
      "ord_total_order_count,ord_total_order_quantity,ord_total_order_amount,ord_cancel_order_count," +
      "ord_refund_order_count,ord_success_first_class_order_count,ord_success_first_class_order_amount," +
      "ord_success_order_type_count,ord_success_self_order_count,ord_success_self_order_amount," +
      "ord_success_aboard_order_count,ord_success_aboard_order_amount,ord_success_order_acity_count," +
      "ord_success_order_cmobile_count,ord_success_flt_last_order_days,ord_success_flt_first_class_order_count," +
      "ord_success_flt_first_class_order_amount,ord_success_flt_max_order_amount,ord_success_flt_avg_order_pricerate," +
      "ord_success_flt_aboard_order_count,ord_success_flt_aboard_order_amount,ord_success_flt_order_count," +
      "ord_success_flt_order_amount,ord_success_flt_order_acity_count,ord_cancel_flt_order_count," +
      "ord_success_htl_last_order_days,ord_success_htl_first_class_order_count,ord_success_htl_first_class_order_amount," +
      "ord_success_htl_max_order_amount,ord_success_htl_order_refund_ratio,ord_success_htl_aboard_order_count," +
      "ord_success_htl_aboard_order_amount,ord_success_htl_guarantee_order_count,ord_success_htl_noshow_order_count," +
      "ord_success_htl_avg_order_pricerate,ord_success_htl_order_count,ord_success_htl_order_amount," +
      "ord_cancel_htl_order_count,ord_refund_htl_order_count,ord_success_pkg_last_order_days," +
      "ord_success_pkg_first_class_order_count,ord_success_pkg_first_class_order_amount,ord_success_pkg_max_order_amount," +
      "ord_success_pkg_aboard_order_count,ord_success_pkg_aboard_order_amount,ord_success_pkg_order_count," +
      "ord_success_pkg_order_amount,ord_cancel_pkg_order_count,ord_refund_pkg_order_count,ord_success_trn_last_order_days," +
      "ord_success_trn_first_class_order_count,ord_success_trn_first_class_order_amount,ord_success_trn_max_order_amount," +
      "ord_success_trn_order_count,ord_success_trn_order_amount,ord_cancel_trn_order_count,ord_refund_trn_order_count," +
      "com_passenger_count,com_idno_count,com_mobile_count,com_has_child,fai_lackbalance,fai_risk,fai_putwrong," +
      "fai_invalid,bil_payord_count,bil_paysord_count,bil_refundord_count,bil_payord_credit_count,bil_payord_debit_count," +
      "bil_paysord_credit_count,bil_paysord_debit_count,bil_ordertype_count,bil_platform_count,cap_tmoney_balance," +
      "cap_wallet_balance,cap_refund_balance,cap_total_balance,cap_withdrow_count,cap_withdrow_amount,acc_loginday_count," +
      "acc_pwd_count,acc_paypwd_count,acc_bindmobile_count,voi_complaint_count,voi_complrefund_count,voi_comment_count," +
      "uid_samemobile,uid_haspaypwd,pro_ismarketing,ord_refund_flt_order_count,score"

    val fields = schemaString.split(",").map(fieldName => StructField(fieldName, DoubleType, true))
    val schema = StructType(fields)

    val noheader = myfile.filter(line => !line.contains("uid_grade"))

    val myfile2 = noheader.map(_.split(",")).map(p => Row.fromSeq(p.map(x => x.toDouble).toSeq))

    val df = sqlContext.createDataFrame(myfile2, schema)

    val end_Time = System.currentTimeMillis() //获取

    println("运行时间： " + (end_Time - start_Time) + "ms")
    df.show()
    df.printSchema()
    return df

  }

  def readHive(table: String, sc: SparkContext): DataFrame = {
    val hiveContext = new HiveContext(sc)
    val df = hiveContext.sql("SELECT * FROM table")
    return df
  }

  /**
    * 删除HDFS目录
    *
    * @param sc
    * @param path
    * @return
    */
  def deleteHdfsPath(sc: SparkContext, path: String) = {
    val hadoopConf = sc.hadoopConfiguration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val hdfsPath = new Path(path)
    if (hdfs.exists(hdfsPath)) {
      //hdfs.delete(hdfsPath, true)
      System.exit(-1)
    }
  }

  /**
    * Concat DataFrame(s)
    *
    * @param data1
    * @param data2
    * @return
    */
  def concatDataFrame(data1: DataFrame, data2: DataFrame*): DataFrame = {
    var resultDF: DataFrame = data1
    for (df <- data2) {
      val rows = resultDF.rdd.zip(df.rdd).map {
        case (rowLeft, rowRight) => Row.fromSeq(rowLeft.toSeq ++ rowRight.toSeq)
      }
      val schema = StructType(resultDF.schema.fields ++ df.schema.fields)
      resultDF = resultDF.sqlContext.createDataFrame(rows, schema)
    }
    resultDF
  }

  /**
    * 输入一个常规DataFrame的数据，返回一个LabeledPiont型的数据
    *
    * @param df
    * @param labelColumn
    * @return
    */
  def dataFrame2LabelPointRDD(df: DataFrame, labelColumn: String) = {

    val featureColumn = df.drop(labelColumn).columns //注意这里
    df.map { row =>
      LabeledPoint(
        row.getAs[Double](labelColumn), // 注意这个地方需要根据前面读取数据的类型更改，Double/Int
        Vectors.dense(row2Array(row, featureColumn))
      )
    }
  }

  /**
    * 输入一个常规的DataFrame的数据，返回一个Vectors RDD
    *
    * @param df
    * @return
    */
  def dataFrame2VectorRDD(df: DataFrame) = {
    val columns = df.columns
    df.map(row => Vectors.dense(row2Array(row, columns)))
  }

  /**
    * 原因是Vectors.dense只支持数组的输入形式
    * 根据columns的列名生成一个列名的数组（Array）
    *
    * @param row
    * @param featureColumn
    * @return
    */
  def row2Array(row: Row, featureColumn: Array[String]) = {
    val array = new Array[Double](featureColumn.length)
    var i = 0
    for (col <- featureColumn) {
      array(i) = row.getAs(col).toString.toDouble;
      i += 1;
    }
    array
  }
}
