//package com.google.cloud.training.dataanalyst.sandiego;

import org.apache.beam.sdk.coders.AvroCoder;
import org.apache.beam.sdk.coders.DefaultCoder;

@DefaultCoder(AvroCoder.class)
public class VariateInfo {
  private String[] fields;

  private enum Field {
    TIMESTAMP, VARIATEVALUE;
  }

  public VariateInfo() {
    // for Avro
  }

  public static VariateInfo newVariateInfo(String line) {
    String[] pieces = line.split(",");
    VariateInfo info = new VariateInfo();
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
