//package com.google.cloud.training.dataanalyst.sandiego;

import org.apache.beam.sdk.coders.AvroCoder;
import org.apache.beam.sdk.coders.DefaultCoder;

@DefaultCoder(AvroCoder.class)
public class RandomVariateInfo {
  private String[] fields;

  private enum Field {
    TIMESTAMP, VARIATEVALUE;
  }

  public RandomVariateInfo() {
    // for Avro
  }

  public static RandomVariateInfo newRandomVariateInfo(String line) {
    String[] pieces = line.split(",");
    RandomVariateInfo info = new RandomVariateInfo();
    info.fields = pieces;
    return info;
  }

  private String get(Field f) {
    return fields[f.ordinal()];
  }

  public String getTimestamp() {
    return fields[Field.TIMESTAMP.ordinal()];
    // return Timestamp.valueOf(fields[Field.TIMESTAMP.ordinal()]).getTime();
  }

  public double getVariate() {
    return Double.parseDouble(get(Field.VARIATEVALUE));
  }
}
