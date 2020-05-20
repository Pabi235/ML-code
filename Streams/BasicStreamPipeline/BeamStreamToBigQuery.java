/*
 * Copyright (C) 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

//package com.google.cloud.training.dataanalyst.sandiego;

import java.util.ArrayList;
import java.util.List;

import org.apache.beam.runners.dataflow.options.DataflowPipelineOptions;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.pubsub.PubsubIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.values.PCollection;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;

/**
 * A dataflow pipeline that pulls from Pub/Sub and writes to BigQuery
 * 
 * @author Pabi
 * Original author vlakshmanan
 *
 */
@SuppressWarnings("serial")
public class StreamRandomVariates {

  public static interface MyOptions extends DataflowPipelineOptions {
    @Description("Also stream to Bigtable?")
    @Default.Boolean(false)
    boolean getBigtable();

    void setBigtable(boolean b);
  }

  public static void main(String[] args) {
    MyOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().as(MyOptions.class);
    options.setStreaming(true);
    Pipeline p = Pipeline.create(options);

    String topic = "projects/" + options.getProject() + "/topics/SimulateNormals";
    String randomVariateTable = options.getProject() + ":playground.NormalVariateData";

    // Build the table schema for the output table.
    List<TableFieldSchema> Tablefields = new ArrayList<>();
    Tablefields.add(new TableFieldSchema().setName("timestamp").setType("TIMESTAMP"));
    Tablefields.add(new TableFieldSchema().setName("RandomValue").setType("FLOAT"));
    TableSchema schema = new TableSchema().setFields(Tablefields);

    PCollection<VariateInfo> StreamRandomVariates = p //
        .apply("GetMessages", PubsubIO.readStrings().fromTopic(topic)) //
        .apply("ExtractData", ParDo.of(new DoFn<String, VariateInfo>() {  //DoFn<InputType,OutputType>
          @ProcessElement
          public void processElement(ProcessContext c) throws Exception {
            String line = c.element();
            c.output(VariateInfo.newVariateInfo(line));
          }
        }));

    //if (options.getBigtable()) {
    //  BigtableHelper.writeToBigtable(StreamRandomVariates, options);
    //}

    StreamRandomVariates.apply("ToBQRow", ParDo.of(new DoFn<VariateInfo, TableRow>() {
      @ProcessElement
      public void processElement(ProcessContext c) throws Exception {
        TableRow row = new TableRow();
        VariateInfo info = c.element();
        row.set("timestamp", info.getTimestamp());
        row.set("RandomValue", info.getVariate());
        c.output(row);
      }
    })) //
        .apply(BigQueryIO.writeTableRows().to(randomVariateTable)//
            .withSchema(schema)//
            .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
            .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED));

    p.run();
  }
}
