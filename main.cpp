#include <iostream>
#include <fstream>
#include <istream>
#include <cassert>
#include <string>
#include <cmath>
#include <set>
#include "csvstream.hpp"

using namespace std;

// Your classifier should be able to compute the log-probability score of
// a post (i.e. a collection of words) given a particular label. To predict
// a label for a new post, it should choose the label that gives the highest
// log-probability score. See the Prediction section.

// able to predict labels from post

class Classifier
{

public:
  void train(csvstream &trainfile, bool debug)
  {
    // A row is a map<string, string>, key = column name, value = datum

    post_count = 0;
    vocab_size = 0;

    if (debug)
    {
      cout << "training data:" << endl;
    }

    // Read file
    read_file(trainfile, debug);

    vocab_size = words.size();

    cout << "trained on " << int(post_count) << " examples" << endl;

    if (debug)
    {
      cout << "vocabulary size = " << vocab_size << endl;
      cout << endl;
    }

    if (debug)
    {
      cout << "classes:" << endl;
      for (const auto &label : labels)
      {
        cout << "  " << label.first << ", " << label.second << " examples, "
             << "log-prior = " << (log((static_cast<double>(label.second) 
             / static_cast<double>(post_count)))) << endl;
      }
      cout << "classifier parameters:" << endl;
      for (const auto &labelWord : words_in_label)
      {
        cout << "  " << labelWord.first.first << ":" << labelWord.first.second
             << ", count = " << labelWord.second << ", log-likelihood = ";
        int count = words_in_label[labelWord.first];
        if (count != 0)
        {
          cout << (log((static_cast<double>(count) /
                        static_cast<double>(labels[labelWord.first.first]))));
          // cout << "CNZ " << count << " " << label << " " << num << " : "
          //<< word << " " << P  << endl; //how likely it is to see
          // word w in posts with label c
        }
        else
        {
          if (words[labelWord.first.second] == 0)
          {
            cout << log((static_cast<double>(1) /
                         static_cast<double>(post_count)));
            // Use when w does not occur anywhere at all in the training set
            //  cout << "CWZ " << num << " " << label << " : "
            //<< word  << " " << P << endl;
          }
          else
          {
            // cout << "blah" << words[word];
            cout << log((static_cast<double>(words[labelWord.first.second]) 
              / static_cast<double>(post_count)));
            // cout << "CZWNZ " << label << " : " << word  << " " << P << endl;
            // Use when w does not occur in posts
            // labeled C but does occur in the training data overall
          }
        }
        cout << endl;
      }
    }
    cout << endl;
  }

  void read_file(csvstream &trainfile, bool debug)
  {
    vector<pair<string, string>> row;
    while (trainfile >> row)
    {
      std::map<string, bool> existingWords;
      std::map<pair<string, string>, bool>
          existingWordsInLabel;
      for (auto &pair : existingWords)
      {
        pair.second = false;
      }
      for (auto &pair : existingWordsInLabel)
      {
        pair.second = false;
      }
      string label;
      if (debug)
      {
        cout << "  label = ";
      }
      for (unsigned int i = 0; i < row.size(); ++i)
      {
        const string &column_name = row[i].first;
        const string &datum = row[i].second;
        if (column_name == "tag" && !debug)
        {
          label = datum;
          labels[datum]++;
        }
        if (column_name == "tag" && debug)
        {
          label = datum;
          labels[datum]++;
          cout << label << ", ";
        }
        if (column_name == "content" && !debug)
        {
          istringstream content(datum);
          read_content(label, content,
                       existingWords, existingWordsInLabel);
        }
        if (column_name == "content" && debug)
        {
          cout << "content = " << datum << endl;
          istringstream content(datum);
          read_content(label, content,
                       existingWords, existingWordsInLabel);
        }
      }
      post_count++;
    }
  }

  void read_content(string label, istringstream &content,
        map<string, bool> &existingWords,
        map<pair<string, string>, bool> &existingWordsInLabel)
  {
    string w;
    while (content >> w)
    {
      if (existingWords[w] == false)
      {
        words[w]++;
        existingWords[w] = true;
      }
      pair<string, string> pair;
      pair.first = label;
      pair.second = w;
      if (existingWordsInLabel[pair] == false)
      {
        words_in_label[pair]++;
        existingWordsInLabel[pair] = true;
      }
    }
  }

  void predict(csvstream &testfile)
  {
    cout << "test data:" << endl;

    int post_amount = 0;
    int correct_amount = 0;
    vector<pair<string, string>> row;
    while (testfile >> row)
    {
      pair<string, double> temp_pair;
      string write_content;
      row_reading(row, correct_amount, temp_pair, write_content);

      string predict_label = temp_pair.first;
      double predict_score = temp_pair.second;

      post_amount++;
      cout << predict_label
           << ", log-probability score = " << predict_score << endl;
      cout << "  "
           << "content = " << write_content << endl
           << endl;
    }
    cout << "performance: " << correct_amount << " / " << post_amount;
    cout << " posts predicted correctly" << endl;
  }

  void row_reading(vector<pair<string, string>> &row, int &correct_amount,
                   pair<string, double> &temp_pair, string &write_content)
  {
    string predict_label;
    double predict_score = 0.0;
    string label;
    for (unsigned int i = 0; i < row.size(); ++i)
    {
      const string &column_name = row[i].first;
      const string &datum = row[i].second;

      if (column_name == "tag" && !datum.empty())
      {
        label = datum;
        cout << "  "
             << "correct = " << label << ", predicted = ";
      }

      if (column_name == "content" && !datum.empty())
      {
        write_content = datum;
        pair<string, double> predict_pair = max_log(datum);
        predict_label = predict_pair.first;
        predict_score = predict_pair.second;
        temp_pair.first = predict_label;
        temp_pair.second = predict_score;

        if (label == predict_label)
        {
          correct_amount++;
        }
      }

      if (column_name == "content" && datum.empty())
      {
        write_content = datum;
        pair<string, double> predict_pair = max_log(datum);
        predict_label = predict_pair.first;
        predict_score = predict_pair.second;
        temp_pair.first = predict_label;
        temp_pair.second = predict_score;

        if (label == predict_label)
        {
          correct_amount++;
        }
      }
    }
  }

  pair<string, double> max_log(string predict_words)
  {
    vector<pair<string, double>> logs;
    set<string> uniques = unique_words(predict_words);

    for (pair<const string, int> &label : labels)
    {
      pair<string, double> p = make_pair(label.first,
        log_probability(label.first, uniques));
      logs.push_back(p);
      // cout << "Probability is " << p.second << endl;
    }

    double max_log_prob = -10000000;
    size_t max_index = 0;
    for (size_t i = 0; i < logs.size(); ++i)
    {
      if (logs[i].second > max_log_prob)
      {
        max_log_prob = logs[i].second;
        max_index = i;
      }
      else if (logs[i].second ==
          max_log_prob &&
          logs[i].first < logs[max_index].first)
      {
        max_log_prob = logs[i].second;
        max_index = i;
      }
    }
    return logs[max_index];
  }

  double log_probability(string label, set<string> input_words)
  {

    int num = labels[label];
    double P = (log((static_cast<double>(num) /
                     (post_count))));
    // cout << label << " init num " << num <<
    //"init post_count " << post_count << " P " << P << endl;
    string word;

    for (auto it = input_words.begin(); it != input_words.end(); ++it)
    {
      // cout << "I'm here" << endl;
      pair<string, string> label_word_pair;
      label_word_pair.first = label;
      label_word_pair.second = *it;
      int count = words_in_label[label_word_pair];
      // Number of occurrences of the word in posts with the given label

      // cout << label_word_pair.first << " " << label_word_pair.second
      //<< " wordsinlabel " << count << endl;

      if (words[*it] == 0)
      {
        P += log(1 / static_cast<double>(post_count));
      }
      else if (words_in_label[make_pair(label, *it)] == 0)
      {
        P += log((static_cast<double>(words[*it]) / (post_count)));
      }
      else
      {
        P += (log((static_cast<double>(count) / (num))));
      }

      // if (count != 0)
      // {
      // P += (log((static_cast<double>(count) / static_cast<double>(num))));
      //   // cout << "CNZ " << count << " " << label << " " << num << " : "
      //   //<< word << " " << P  << endl; //how likely it is to see word w
      //   // in posts with label c
      // }
      // else
      // {
      //   if (words[word] == 0)
      //   {
      //  P += log((static_cast<double>(1) / static_cast<double>(post_count)));
      //     // Use when w does not occur anywhere at all in the training set
      //     //  cout << "CWZ " << num << " " << label << " : "
      //     //<< word  << " " << P << endl;
      //   }
      //   else
      //   {
      //     // cout << "blah" << words[word];
      //     P += log((static_cast<double>(words[word])
      //     / static_cast<double>(post_count)));
      // cout << "CZWNZ " << label << " : " << word  << " " << P << endl;
      // Use when w does not occur in posts labeled C
      // but does occur in the training data overall
      //   }
      // }
    }
    return P;
  }

  // EFFECTS: Return a set of unique whitespace delimited words.x
  set<string> unique_words(const string &str)
  {
    istringstream source(str);
    set<string> words;
    string word;
    while (source >> word)
    {
      words.insert(word);
    }
    return words;
  }

private:
  double post_count;
  int vocab_size;
  std::map<string, int> words;
  std::map<string, int> labels;
  std::map<pair<string, string>, int> words_in_label;
};

int main(int argc, char **argv)
{
  cout.precision(3);
  bool debug = false;
  if (argc == 3)
  {
  }
  else if (argc == 4)
  {
    if (string(argv[3]) == "--debug")
    {
      debug = true;
    }
    else
    {
      cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
      return 1;
    }
  }
  else
  {
    cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
    return 1;
  }

  string file_name = string(argv[1]);
  csvstream file(file_name);
  if (!file)
  {
    cout << "Error opening file: " << file_name << endl;
    return 1;
  }

  string test_file_name = string(argv[2]);
  csvstream test_file(test_file_name);
  if (!test_file)
  {
    cout << "Error opening file: " << test_file_name << endl;
    return 1;
  }

  Classifier c;
  c.train(file, debug);
  c.predict(test_file);

  return 0;
}
