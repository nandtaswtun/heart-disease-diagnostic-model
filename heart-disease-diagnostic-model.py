# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

# %%
df = pd.read_csv('C:\\Users\\Lenovo\\Downloads\\Heart_Disease_Prediction.csv')
df.head()

# %%
cholesterol = df.groupby('Heart Disease')['Cholesterol'].mean().reset_index()

# %%
print(f' --- Mean Cholesterol Levels by Heart Disease Status --- ')
print(cholesterol)

# %%
df['Target'] = df['Heart Disease'].map({'Presence' : 1, 'Absence' : 0})

# %%
plt.figure(figsize = (10, 6))
sns.scatterplot(data = df, x = 'Age', y = 'Cholesterol', hue = 'Heart Disease')
plt.title('Age vs Cholesterol by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# %%
x = df.drop(columns = ['Heart Disease', 'Target'])
y = df['Target']

# %%
t_stat, p_value = stats.ttest_ind(df[df['Heart Disease'] == 'Presence']['Cholesterol'],
                                    df[df['Heart Disease'] == 'Absence']['Cholesterol'])
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# %%
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# %%
model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"**Model Accuracy:** {accuracy:.2%}")


# %%
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'R-squared: {r2}')
print(f'Mean Absolute Error: {mae}')

# %%
cm = confusion_matrix(y_test, model.predict(x_test))
print(cm)

# %%
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
report = classification_report(y_test, model.predict(x_test))
print(f'--- \n{report}\n ---')


