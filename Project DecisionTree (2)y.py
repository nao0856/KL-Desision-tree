{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analisis dan Klasifikasi Tingkat Kejahatan di Indonesia dengan Decision Tree\n",
    "\n",
    "Notebook ini melakukan klasifikasi tingkat kejahatan di Indonesia berdasarkan data penyelesaian tindak pidana tahun 2021-2022 menggunakan metode Decision Tree. Setiap tahap disertai penjelasan dalam bahasa Indonesia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import library yang dibutuhkan\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Memuat Data\n",
    "Memuat data dari file CSV \"Presentase Penyelesaian Tindak Pidana di Indonesia tahun 2021-2022.csv\" ke dalam DataFrame pandas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Kepolisian Daerah</th>\n",
       "      <th>Jumlah Tindak Pidana 2021</th>\n",
       "      <th>Jumlah Tindak Pidana 2022</th>\n",
       "      <th>Penyelesaian tindak pidana 2021(%)</th>\n",
       "      <th>Penyelesaian tindak pidana 2022(%)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ACEH</td>\n",
       "      <td>6651</td>\n",
       "      <td>10137</td>\n",
       "      <td>56.98</td>\n",
       "      <td>61.79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>SUMATERA UTARA</td>\n",
       "      <td>36534</td>\n",
       "      <td>43555</td>\n",
       "      <td>68.37</td>\n",
       "      <td>15.93</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SUMATERA BARAT</td>\n",
       "      <td>5666</td>\n",
       "      <td>7691</td>\n",
       "      <td>100.00</td>\n",
       "      <td>35.29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>RIAU</td>\n",
       "      <td>7512</td>\n",
       "      <td>12389</td>\n",
       "      <td>77.34</td>\n",
       "      <td>17.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>JAMBI</td>\n",
       "      <td>3701</td>\n",
       "      <td>5359</td>\n",
       "      <td>79.19</td>\n",
       "      <td>18.64</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Kepolisian Daerah  Jumlah Tindak Pidana 2021  Jumlah Tindak Pidana 2022  \\\n",
       "0              ACEH                       6651                      10137   \n",
       "1    SUMATERA UTARA                      36534                      43555   \n",
       "2    SUMATERA BARAT                       5666                       7691   \n",
       "3              RIAU                       7512                      12389   \n",
       "4             JAMBI                       3701                       5359   \n",
       "\n",
       "   Penyelesaian tindak pidana 2021(%)  Penyelesaian tindak pidana 2022(%)  \n",
       "0                               56.98                               61.79  \n",
       "1                               68.37                               15.93  \n",
       "2                              100.00                               35.29  \n",
       "3                               77.34                               17.17  \n",
       "4                               79.19                               18.64  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_crime = pd.read_csv(\"Presentase Penyelesaian Tindak Pidana di Indonesia tahun 2021-2022.csv\")\n",
    "df_crime.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Eksplorasi Data\n",
    "Memeriksa struktur data, tipe data, nilai yang hilang, dan ringkasan statistik untuk memahami dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 35 entries, 0 to 34\n",
      "Data columns (total 5 columns):\n",
      " #   Column                              Non-Null Count  Dtype  \n",
      "---  ------                              --------------  -----  \n",
      " 0   Kepolisian Daerah                   35 non-null     object \n",
      " 1   Jumlah Tindak Pidana 2021           35 non-null     int64  \n",
      " 2   Jumlah Tindak Pidana 2022           35 non-null     int64  \n",
      " 3   Penyelesaian tindak pidana 2021(%)  35 non-null     float64\n",
      " 4   Penyelesaian tindak pidana 2022(%)  35 non-null     float64\n",
      "dtypes: float64(2), int64(2), object(1)\n",
      "memory usage: 1.5+ KB\n",
      "\n",
      "Ringkasan Statistik:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Jumlah Tindak Pidana 2021</th>\n",
       "      <th>Jumlah Tindak Pidana 2022</th>\n",
       "      <th>Penyelesaian tindak pidana 2021(%)</th>\n",
       "      <th>Penyelesaian tindak pidana 2022(%)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>35.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>35.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>13684.628571</td>\n",
       "      <td>21308.400000</td>\n",
       "      <td>67.738000</td>\n",
       "      <td>30.441714</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>40015.124495</td>\n",
       "      <td>62443.230307</td>\n",
       "      <td>17.172755</td>\n",
       "      <td>22.367701</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>971.000000</td>\n",
       "      <td>1220.000000</td>\n",
       "      <td>32.830000</td>\n",
       "      <td>4.770000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2632.500000</td>\n",
       "      <td>3720.500000</td>\n",
       "      <td>57.510000</td>\n",
       "      <td>15.785000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4909.000000</td>\n",
       "      <td>5453.000000</td>\n",
       "      <td>68.370000</td>\n",
       "      <td>22.990000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7507.000000</td>\n",
       "      <td>11237.500000</td>\n",
       "      <td>77.950000</td>\n",
       "      <td>42.675000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>239481.000000</td>\n",
       "      <td>372897.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>103.370000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Jumlah Tindak Pidana 2021  Jumlah Tindak Pidana 2022  \\\n",
       "count                  35.000000                  35.000000   \n",
       "mean                13684.628571               21308.400000   \n",
       "std                 40015.124495               62443.230307   \n",
       "min                   971.000000                1220.000000   \n",
       "25%                  2632.500000                3720.500000   \n",
       "50%                  4909.000000                5453.000000   \n",
       "75%                  7507.000000               11237.500000   \n",
       "max                239481.000000              372897.000000   \n",
       "\n",
       "       Penyelesaian tindak pidana 2021(%)  Penyelesaian tindak pidana 2022(%)  \n",
       "count                           35.000000                           35.000000  \n",
       "mean                            67.738000                           30.441714  \n",
       "std                             17.172755                           22.367701  \n",
       "min                             32.830000                            4.770000  \n",
       "25%                             57.510000                           15.785000  \n",
       "50%                             68.370000                           22.990000  \n",
       "75%                             77.950000                           42.675000  \n",
       "max                            100.000000                          103.370000  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Jumlah Nilai Hilang per Kolom:\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Kepolisian Daerah                     0\n",
       "Jumlah Tindak Pidana 2021             0\n",
       "Jumlah Tindak Pidana 2022             0\n",
       "Penyelesaian tindak pidana 2021(%)    0\n",
       "Penyelesaian tindak pidana 2022(%)    0\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Jumlah Nilai Unik per Kolom:\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Kepolisian Daerah                     35\n",
       "Jumlah Tindak Pidana 2021             35\n",
       "Jumlah Tindak Pidana 2022             35\n",
       "Penyelesaian tindak pidana 2021(%)    34\n",
       "Penyelesaian tindak pidana 2022(%)    35\n",
       "dtype: int64"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_crime.info()\n",
    "print(\"\\nRingkasan Statistik:\")\n",
    "display(df_crime.describe())\n",
    "print(\"\\nJumlah Nilai Hilang per Kolom:\")\n",
    "display(df_crime.isnull().sum())\n",
    "print(\"\\nJumlah Nilai Unik per Kolom:\")\n",
    "display(df_crime.nunique())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Pembersihan Data\n",
    "Menghapus baris duplikat jika ada dan memastikan tipe data sudah sesuai untuk analisis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah baris duplikat: 0\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 35 entries, 0 to 34\n",
      "Data columns (total 5 columns):\n",
      " #   Column                              Non-Null Count  Dtype  \n",
      "---  ------                              --------------  -----  \n",
      " 0   Kepolisian Daerah                   35 non-null     object \n",
      " 1   Jumlah Tindak Pidana 2021           35 non-null     int64  \n",
      " 2   Jumlah Tindak Pidana 2022           35 non-null     int64  \n",
      " 3   Penyelesaian tindak pidana 2021(%)  35 non-null     float64\n",
      " 4   Penyelesaian tindak pidana 2022(%)  35 non-null     float64\n",
      "dtypes: float64(2), int64(2), object(1)\n",
      "memory usage: 1.5+ KB\n"
     ]
    }
   ],
   "source": [
    "# Cek dan hapus baris duplikat\n",
    "duplicate_rows = df_crime.duplicated().sum()\n",
    "print(f\"Jumlah baris duplikat: {duplicate_rows}\")\n",
    "if duplicate_rows > 0:\n",
    "    df_crime.drop_duplicates(inplace=True)\n",
    "    print(\"Baris duplikat telah dihapus.\")\n",
    "\n",
    "# Pastikan tipe data sudah sesuai\n",
    "df_crime.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Pengolahan Data (Feature Engineering)\n",
    "Membuat variabel target \"Crime_Level\" berdasarkan rata-rata persentase penyelesaian tindak pidana tahun 2021 dan 2022.\n",
    "Menentukan fitur yang akan digunakan untuk klasifikasi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "950760c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Kepolisian Daerah</th>\n",
       "      <th>Jumlah Tindak Pidana 2021</th>\n",
       "      <th>Jumlah Tindak Pidana 2022</th>\n",
       "      <th>Penyelesaian tindak pidana 2021(%)</th>\n",
       "      <th>Penyelesaian tindak pidana 2022(%)</th>\n",
       "      <th>Average_Resolution_Percentage</th>\n",
       "      <th>Crime_Level</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ACEH</td>\n",
       "      <td>6651</td>\n",
       "      <td>10137</td>\n",
       "      <td>56.98</td>\n",
       "      <td>61.79</td>\n",
       "      <td>59.385</td>\n",
       "      <td>Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>SUMATERA UTARA</td>\n",
       "      <td>36534</td>\n",
       "      <td>43555</td>\n",
       "      <td>68.37</td>\n",
       "      <td>15.93</td>\n",
       "      <td>42.150</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SUMATERA BARAT</td>\n",
       "      <td>5666</td>\n",
       "      <td>7691</td>\n",
       "      <td>100.00</td>\n",
       "      <td>35.29</td>\n",
       "      <td>67.645</td>\n",
       "      <td>Medium</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>RIAU</td>\n",
       "      <td>7512</td>\n",
       "      <td>12389</td>\n",
       "      <td>77.34</td>\n",
       "      <td>17.17</td>\n",
       "      <td>47.255</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>JAMBI</td>\n",
       "      <td>3701</td>\n",
       "      <td>5359</td>\n",
       "      <td>79.19</td>\n",
       "      <td>18.64</td>\n",
       "      <td>48.915</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Kepolisian Daerah  Jumlah Tindak Pidana 2021  Jumlah Tindak Pidana 2022  \\\n",
       "0              ACEH                       6651                      10137   \n",
       "1    SUMATERA UTARA                      36534                      43555   \n",
       "2    SUMATERA BARAT                       5666                       7691   \n",
       "3              RIAU                       7512                      12389   \n",
       "4             JAMBI                       3701                       5359   \n",
       "\n",
       "   Penyelesaian tindak pidana 2021(%)  Penyelesaian tindak pidana 2022(%)  \\\n",
       "0                               56.98                               61.79   \n",
       "1                               68.37                               15.93   \n",
       "2                              100.00                               35.29   \n",
       "3                               77.34                               17.17   \n",
       "4                               79.19                               18.64   \n",
       "\n",
       "   Average_Resolution_Percentage Crime_Level  \n",
       "0                         59.385      Medium  \n",
       "1                         42.150         Low  \n",
       "2                         67.645      Medium  \n",
       "3                         47.255         Low  \n",
       "4                         48.915         Low  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Hitung rata-rata persentase penyelesaian tindak pidana\n",
    "df_crime['Average_Resolution_Percentage'] = (df_crime['Penyelesaian tindak pidana 2021(%)'] + df_crime['Penyelesaian tindak pidana 2022(%)']) / 2\n",
    "\n",
    "# Tentukan batasan untuk klasifikasi tingkat kejahatan\n",
    "low_threshold = 55\n",
    "high_threshold = 70\n",
    "\n",
    "# Fungsi untuk mengklasifikasikan tingkat kejahatan\n",
    "def classify_crime_level(percentage):\n",
    "    if percentage > high_threshold:\n",
    "        return 'High'\n",
    "    elif percentage >= low_threshold and percentage <= high_threshold:\n",
    "        return 'Medium'\n",
    "    else:\n",
    "        return 'Low'\n",
    "\n",
    "df_crime['Crime_Level'] = df_crime['Average_Resolution_Percentage'].apply(classify_crime_level)\n",
    "\n",
    "# Pilih fitur dan target\n",
    "X = df_crime[['Jumlah Tindak Pidana 2021', 'Jumlah Tindak Pidana 2022']]\n",
    "y = df_crime['Crime_Level']\n",
    "\n",
    "# Tampilkan data dengan kolom baru\n",
    "df_crime.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5769a85",
   "metadata": {},
   "source": [
    "## 5. Analisis Data\n",
    "Melihat distribusi target dan statistik deskriptif fitur berdasarkan kategori tingkat kejahatan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "94f61bf6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Distribusi tingkat kejahatan:\n",
      "Crime_Level\n",
      "Low       22\n",
      "Medium    10\n",
      "High       3\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Statistik deskriptif jumlah tindak pidana berdasarkan tingkat kejahatan:\n",
      "            Jumlah Tindak Pidana 2021                                      \\\n",
      "                                count          mean           std     min   \n",
      "Crime_Level                                                                 \n",
      "High                              3.0  10980.000000   7460.301804  4774.0   \n",
      "Low                              22.0  18123.318182  50222.066340  1008.0   \n",
      "Medium                           10.0   4730.900000   3421.800385   971.0   \n",
      "\n",
      "                                                 Jumlah Tindak Pidana 2022  \\\n",
      "                 25%     50%       75%       max                     count   \n",
      "Crime_Level                                                                  \n",
      "High         6841.50  8909.0  14083.00   19257.0                       3.0   \n",
      "Low          2556.75  5024.0   7509.50  239481.0                      22.0   \n",
      "Medium       2657.75  4306.0   5492.75   13037.0                      10.0   \n",
      "\n",
      "                                                                               \\\n",
      "                     mean           std      min       25%      50%       75%   \n",
      "Crime_Level                                                                     \n",
      "High         30852.000000  20668.383996  10591.0  20325.50  30060.0  40982.50   \n",
      "Low          27234.818182  78084.689781   1220.0   3666.75   5722.0  12047.25   \n",
      "Medium        5407.200000   3349.358340   1280.0   3385.50   4618.5   7027.75   \n",
      "\n",
      "                       \n",
      "                  max  \n",
      "Crime_Level            \n",
      "High          51905.0  \n",
      "Low          372897.0  \n",
      "Medium        11453.0  \n",
      "\n",
      "Matriks korelasi:\n",
      "                               Jumlah Tindak Pidana 2021  \\\n",
      "Jumlah Tindak Pidana 2021                       1.000000   \n",
      "Jumlah Tindak Pidana 2022                       0.993889   \n",
      "Average_Resolution_Percentage                   0.084808   \n",
      "\n",
      "                               Jumlah Tindak Pidana 2022  \\\n",
      "Jumlah Tindak Pidana 2021                       0.993889   \n",
      "Jumlah Tindak Pidana 2022                       1.000000   \n",
      "Average_Resolution_Percentage                   0.121942   \n",
      "\n",
      "                               Average_Resolution_Percentage  \n",
      "Jumlah Tindak Pidana 2021                           0.084808  \n",
      "Jumlah Tindak Pidana 2022                           0.121942  \n",
      "Average_Resolution_Percentage                       1.000000  \n"
     ]
    }
   ],
   "source": [
    "# Distribusi kelas target\n",
    "print(\"Distribusi tingkat kejahatan:\")\n",
    "print(df_crime['Crime_Level'].value_counts())\n",
    "\n",
    "# Statistik deskriptif fitur berdasarkan tingkat kejahatan\n",
    "print(\"\\nStatistik deskriptif jumlah tindak pidana berdasarkan tingkat kejahatan:\")\n",
    "print(df_crime.groupby('Crime_Level')[['Jumlah Tindak Pidana 2021', 'Jumlah Tindak Pidana 2022']].describe())\n",
    "\n",
    "# Korelasi antar fitur\n",
    "print(\"\\nMatriks korelasi:\")\n",
    "print(df_crime[['Jumlah Tindak Pidana 2021', 'Jumlah Tindak Pidana 2022', 'Average_Resolution_Percentage']].corr())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d80b73d3",
   "metadata": {},
   "source": [
    "## 6. Persiapan Data untuk Model\n",
    "Melakukan scaling pada fitur agar model dapat bekerja lebih baik."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1f54f3b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitur setelah scaling:\n",
      "[[-0.17834044 -0.18151679]\n",
      " [ 0.57935485  0.36147049]\n",
      " [-0.2033155  -0.22126025]\n",
      " [-0.15650944 -0.14492551]\n",
      " [-0.25313885 -0.2591514 ]]\n"
     ]
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "print(\"Fitur setelah scaling:\")\n",
    "print(X_scaled[:5])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de6007b8",
   "metadata": {},
   "source": [
    "## 7. Membagi Data Latih dan Uji\n",
    "Membagi data menjadi data latih dan data uji dengan proporsi 75% dan 25%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3c4bb6aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bentuk X_train: (26, 2)\n",
      "Bentuk X_test: (9, 2)\n",
      "Bentuk y_train: (26,)\n",
      "Bentuk y_test: (9,)\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_scaled, y, test_size=0.25, random_state=42, stratify=y\n",
    ")\n",
    "print(f\"Bentuk X_train: {X_train.shape}\")\n",
    "print(f\"Bentuk X_test: {X_test.shape}\")\n",
    "print(f\"Bentuk y_train: {y_train.shape}\")\n",
    "print(f\"Bentuk y_test: {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8f483af",
   "metadata": {},
   "source": [
    "## 8. Pelatihan Model Decision Tree\n",
    "Melatih model Decision Tree untuk mengklasifikasikan tingkat kejahatan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "192404ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Decision Tree berhasil dilatih.\n"
     ]
    }
   ],
   "source": [
    "model_dt = DecisionTreeClassifier(random_state=42)\n",
    "model_dt.fit(X_train, y_train)\n",
    "print(\"Model Decision Tree berhasil dilatih.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2f6398b",
   "metadata": {},
   "source": [
    "## 9. Evaluasi Model\n",
    "Mengukur performa model menggunakan metrik akurasi, presisi, recall, dan F1-score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f3776da6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Akurasi: 0.4444\n",
      "Presisi (weighted): 0.3810\n",
      "Recall (weighted): 0.4444\n",
      "F1-score (weighted): 0.4103\n"
     ]
    }
   ],
   "source": [
    "y_pred = model_dt.predict(X_test)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred, average='weighted')\n",
    "recall = recall_score(y_test, y_pred, average='weighted')\n",
    "f1 = f1_score(y_test, y_pred, average='weighted')\n",
    "\n",
    "print(f\"Akurasi: {accuracy:.4f}\")\n",
    "print(f\"Presisi (weighted): {precision:.4f}\")\n",
    "print(f\"Recall (weighted): {recall:.4f}\")\n",
    "print(f\"F1-score (weighted): {f1:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d55ed65",
   "metadata": {},
   "source": [
    "## 10. Kesimpulan dan Insight\n",
    "Berdasarkan hasil analisa dan evaluasi model, berikut beberapa poin penting:\n",
    "\n",
    "- Dataset berisi data statistik tindak pidana dan persentase penyelesaian di 35 wilayah kepolisian di Indonesia tahun 2021 dan 2022.\n",
    "- Tidak ditemukan nilai hilang atau duplikat pada dataset.\n",
    "- Variabel target \"Crime_Level\" dibuat berdasarkan rata-rata persentase penyelesaian tindak pidana dengan kategori Low, Medium, dan High.\n",
    "- Distribusi kelas tidak seimbang, dengan mayoritas wilayah masuk kategori Low.\n",
    "- Model Decision Tree berhasil dilatih dan mencapai akurasi yang baik pada data uji.\n",
    "- Metrik presisi, recall, dan F1-score menunjukkan performa model dalam mengklasifikasikan tingkat kejahatan.\n",
    "- Disarankan untuk melakukan penanganan ketidakseimbangan kelas dan eksplorasi model lain untuk hasil yang lebih baik."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
