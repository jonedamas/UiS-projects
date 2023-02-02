import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# BET plot y-value transform
def y_val(p, V):
    return 1/(V*((1/p)-1))

# Rouquerol y-value transform
def r_val(p, V):
    return (V*(1 - p))

# Sample class
class BETSample:
    def __init__(self, samplename, file, des=False):
        self.samplename = samplename
        self.des = des
        
        # Dataframe
        self.df = pd.read_csv(file)
        self.dfPA = self.df['Pressure'].dropna()
        self.dfA = self.df['Quantity'].dropna()
        if des:
            self.dfPD = self.df['PresDes'].dropna()
            self.dfD = self.df['Des'].dropna()

st.title('BET Analysis')
st.markdown('BET equation:')
st.latex(r'''\color{red}{\frac{1}{V(\frac{P°}{P}-1)}}=\color{blue}{\frac{C-1}{V_{m}\cdot C}}\Biggl(\frac{P}{P°}\Biggr) +\color{orange}{\frac{1}{V_{m}\cdot C}}''')
st.markdown('The values for the $\color{red}{red}$ expression is calculated from the quantity adsorbed and the inverse relative pressure of the different data poits, and is plotted on the y-axis in the BET plot.\
 The $Black$ expression is the relative pressure, and is plotted on the x-axis.\
The values for the $\color{blue}{blue}$ and $\color{orange}{orange}$ expressions can then be found by applying a linear regression, yielding the slope and intercept respectively. Monolayer pore volume and BET constant is calculated by the slope and intercept from the linear regression by the following equations derived from the BET equation:')

file = st.file_uploader('Choose a CSV file for BET Analysis')

if file is not None:
    sp = BETSample('sample', file)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Adsorption isotherm
    ax.plot(sp.dfPA, sp.dfA, color='red', label='Adsorption')
    ax.scatter(sp.dfPA, sp.dfA, marker='.', color='red')
        
    ax.set_xlabel('P/P$^o$', fontsize=13)
    ax.set_ylabel('Quantity adsorbed [cm$^3$/g STP]', fontsize=12)
    ax.set_title(f'77K N$_{2}$ Isotherm plot', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid();

    bet_r = list(r_val(sp.dfPA, sp.dfA))
    maxpoint = max(bet_r)
    maxloc = sp.dfPA[bet_r.index(maxpoint)]

    print(f'Data points for BET calculations should not exceed a P/P° of {maxloc:.4f}.')

    ax2.scatter(sp.dfPA, bet_r, facecolors='none', edgecolors='dodgerblue')
    ax2.scatter(maxloc, maxpoint, color='red')
    ax2.vlines(x=maxloc, ymin=0, ymax=max(bet_r), color='crimson', linestyle='--')

    ax2.set_title(f'Rouquerol plot', fontsize=13)
    ax2.set_xlabel('P/P°', fontsize=13)
    ax2.set_ylabel('V(1-P/P°)', fontsize=12)
    ax2.grid();

    st.pyplot(fig)
    st.text(f'Recomended upper BET range limit: {maxloc:.4f}')

    fig2, ax2 = plt.subplots(1)

    col1, col2 = st.columns(2)
    lower_range = col1.number_input('Lower BET range', step=0.001, value=0.01, min_value=0.001, help='Lower limit of the adsorption data points to be used for BET calculations')
    upper_range = col2.number_input('Upper BET range', step=0.001, value=maxloc, help='Upper limit of the adsorption data points to be used for BET calculations')

    BET_range = [lower_range, upper_range]

    Vpoint = 0.983

    ix_low = np.where(sp.dfPA==min([i for i in sp.dfPA.tolist() if i < BET_range[0]], key=lambda x:abs(x-BET_range[0])))[0][0]

    ix_high = np.where(sp.dfPA==min([i for i in sp.dfPA.tolist() if i >= BET_range[1]], key=lambda x:abs(x-BET_range[1])))[0][0]
    
    bet_p = sp.dfPA[ix_low:ix_high].values.reshape(-1,1)
    bet_q = sp.dfA[ix_low:ix_high].values.reshape(-1,1)
    bet_y = y_val(bet_p, bet_q)

    # adsorbed amount at chosen relative pressure for total volume:
    ix_point = min(range(len(sp.dfPA)), key = lambda i: abs(sp.dfPA[i]-Vpoint))
    load = sp.dfA[ix_point]

    # Creating a linear regression model
    model = LinearRegression().fit(bet_p, bet_y)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])
    mlpv = 1/(slope + intercept)
    C = 1 + slope / intercept
    R2 = model.score(bet_p, bet_y)

    if C > 0:
        st.warning('C-value is positive' ,icon='⚠️')

    d = {'R2':R2, 'C':C, 'Monolayer pore volume':round(mlpv, 2), 'BET surface area':round(4.36 * mlpv, 2), 'Total pore volume':load * 1.547e-3}
    df = pd.DataFrame(data=d, index=[0])

    # Plotting regression line and extracted data points:
    x_val = np.linspace(bet_p[0], bet_p[-1], 100)
    ax2.plot(x_val, x_val * slope + intercept,':' , color='red')
    ax2.scatter(bet_p, bet_y, marker='p', color='red')
    ax2.set_xlabel('P/P$^o$', fontsize=13)
    ax2.set_ylabel('1/V(P$^o$/P-1)', fontsize=12)
    ax2.set_title(f'BET plot - {sp.samplename}', fontsize=13)
    ax2.text(bet_p[0], bet_y[-1], f'R$^2$: {R2:.4f}', fontsize=13, color='red')

    # Plot axis limits
    x_factor, y_factor = (bet_p[-1] - bet_p[0]) * 0.2, (bet_y[-1] - bet_y[0]) * 0.2
    ax2.set_xlim((bet_p[0] - x_factor,  bet_p[-1] + x_factor))
    ax2.set_ylim((bet_y[0] - y_factor,  bet_y[-1] + y_factor))
    ax2.grid();

    st.table(df)
    st.pyplot(fig2)