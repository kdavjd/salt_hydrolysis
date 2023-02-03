import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from functions.data_functions import DataFunctions as dfc
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid


plt.style.use(['science', 'no-latex', 'notebook', 'grid'])
    
    
class Plot():
    
    def figure_1(df):    
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,15))

        #график выхода анатаза
        x = df[df['анатаз'].notna()].sort_values(by='температура')['температура']
        y = df[df['анатаз'].notna()].sort_values(by='температура')['анатаз']
        axes[0].scatter(x, y)
        axes[0].plot(x, dfc.get_fit(x,y, n=3), 'o--', color='red', lw=2, ms=3)
        axes[0].set_ylabel('$ \omega_{анатаз}, \%$', fontsize=15)
        axes[0].set_xlabel('$t, °C$', fontsize=15)

        #график выхода брукита
        x = df[df['брукит'].notna()].sort_values(by='температура')['температура']
        y = df[df['брукит'].notna()].sort_values(by='температура')['брукит']
        x_ = np.linspace(min(x), 161, 100)
        y_ = dfc.get_coeffs(x,y,n=4)
        axes[1].scatter(x, y)
        axes[1].plot(x_, np.polyval(y_,x_), 'o--', color='red', lw=2, ms=0.5)
        axes[1].set_ylabel('$ \omega_{брукит}, \%$', fontsize=15)
        axes[1].set_xlabel('$t, °C$', fontsize=15)

        #график выхода рутила
        x = df[df['рутил'].notna()].sort_values(by='температура')['температура']
        y = df[df['рутил'].notna()].sort_values(by='температура')['рутил']
        x_ = np.linspace(min(x), max(x), 100)
        y_ = dfc.get_coeffs(x,y,n=3)
        axes[2].scatter(x, y)
        axes[2].plot(x_, np.polyval(y_,x_), 'o--', color='red', lw=2, ms=0.5)
        axes[2].set_ylabel('$ \omega_{рутил}, \%$', fontsize=15)
        axes[2].set_xlabel('$t, °C$', fontsize=15)    
        
        return fig
    
    def figure_2(df):
        
        fig, (ax,ax2,ax3) = plt.subplots(1, 3, figsize=(10, 6))

        df1= df.pivot_table(index='температура',columns='область_концентраций',values='анатаз').drop(142)
        df1 = df1[['минимальная','малая','средняя','высокая']]
        df1.columns=[5,10,15,20]
        im = ax.imshow(df1.values)
        ax.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax.set_ylabel(r'$ t,°C $',fontdict={'size':12})
        ax.set_xlabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $',fontdict={'size':12})
        ax.set_title('Анатаз',fontdict={'size':12})
        ax.tick_params(axis='both', which='major', labelsize=9)

        df1= df.pivot_table(index='температура',columns='область_концентраций',values='брукит').drop(142)
        df1 = df1[['минимальная','малая','средняя','высокая']]
        df1.columns=[5,10,15,20]
        im = ax2.imshow(df1.values)
        ax2.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax2.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax2.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
        ax2.set_title('Брукит',fontdict={'size':12})
        ax2.set_xlabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $',fontdict={'size':12})
        ax2.tick_params(axis='both', which='major', labelsize=9)

        df1= df.pivot_table(index='температура',columns='область_концентраций',values='рутил').drop(142)
        df1 = df1[['минимальная','малая','средняя','высокая']]
        df1.columns=[5,10,15,20]
        im = ax3.imshow(df1.values)
        ax3.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax3.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax3.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
        ax3.set_title('Рутил',fontdict={'size':12})
        ax3.set_xlabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $',fontdict={'size':12})
        ax3.tick_params(axis='both', which='major', labelsize=9)
        
        return fig
    
    def figure_3(df):
        
        fig = plt.figure(figsize=(12, 15))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        #график насыпной плотности от концентрации
        ax = fig.add_subplot(3,2,1)
        df1= df.pivot_table(index='температура',columns='область_концентраций',values='насыпной_вес')
        x = df1['малая'].dropna().index
        y = df1['малая'].dropna().values
        ax.scatter(x,y)
        y_linear = np.polyval(np.polyfit(x,y,1), x)
        ax.plot(x,y_linear, 'r-', label='$ C_{TiCl_{4}}~11.5\% $')

        _x = df1['высокая'].dropna().index
        _y = df1['высокая'].dropna().values
        _y_linear = np.polyval(np.polyfit(_x,_y,1), _x)
        ax.plot(_x,_y_linear, 'g-', label='$ C_{TiCl_{4}}~20\% $')
        ax.scatter(_x,_y, c='#7f7f7f')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel("$ Насыпная~плотность,\\frac{г}{см^{3}} $", fontdict={'size':12})
        ax.set_xlabel(r'$ t,°C $', fontdict={'size':12})
        ax.legend(loc='upper right', fontsize='large')

        #второй график
        ax2 = fig.add_subplot(3,2,2)
        t150_filter = df['температура'] == 150
        t160_filter = df['температура'] == 160
        low_ratio_filter = (df['соотношение'] <= 0.12) & (df['соотношение'] >= 0.09)
        high_ratio_filter = (df['соотношение'] <= 0.22) & (df['соотношение'] >= 0.19)
        y = df[t150_filter & high_ratio_filter].sort_values(by='концентрация')[['концентрация','агрегаты_размер']].dropna()['агрегаты_размер']
        x = df[t150_filter & high_ratio_filter].sort_values(by='концентрация')[['концентрация','агрегаты_размер']].dropna()['концентрация']
        y_linear = np.polyval(np.polyfit(x,y,1), x)
        ax2.plot(x,y_linear, 'r-', label='$ t,°C=150 $\n$ \\frac{TiO_{2}}{MgCl_{2}}=0.2 $')#\
        ax2.scatter(x,y)

        _y = df[t160_filter & low_ratio_filter].sort_values(by='концентрация')[['концентрация','агрегаты_размер']].dropna()['агрегаты_размер']
        _x = df[t160_filter & low_ratio_filter].sort_values(by='концентрация')[['концентрация','агрегаты_размер']].dropna()['концентрация']
        _y_linear = np.polyval(np.polyfit(_x,_y,1), _x)
        ax2.plot(_x,_y_linear, 'g-', label='$ t,°C=160 $\n$ \\frac{TiO_{2}}{MgCl_{2}}=0.1 $')
        ax2.scatter(_x,_y, c='#7f7f7f')
        ax2.set_xlabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $', fontdict={'size':12})
        ax2.set_ylabel("d50, мкм", fontdict={'size':12})
        ax2.legend(loc='upper right', fontsize='large')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        #третий график
        ax3 = fig.add_subplot(3,2,3)
        x = df[['температура','насыпной_вес']].groupby(by='температура').mean().dropna().index.values
        y = df[['температура','насыпной_вес']].groupby(by='температура').mean()['насыпной_вес'].dropna().values
        plt.scatter(x,y)
        y_linear = np.polyval(np.polyfit(x,y,1), x)
        ax3.plot(x,y_linear, 'r-')
        ax3.set_ylabel("$ Насыпная~плотность,\\frac{г}{см^{3}} $", fontdict={'size':12})
        ax3.set_xlabel(r'$ t,°C $', fontdict={'size':12})
        ax3.tick_params(axis='both', which='major', labelsize=12)

        #четвёртый график
        ax4 = fig.add_subplot(3,2,4)
        def double_exp_func(x, a, b, c, d):
            return a * np.exp(b * x) + c * np.exp(d * x)

        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        x = df[['температура','агрегаты_размер']].groupby(by='температура').mean().dropna().index.values
        y = df[['температура','агрегаты_размер']].groupby(by='температура').mean()['агрегаты_размер'].dropna().values

        ax4.scatter(x,y)
        popt, pcov = curve_fit(exp_func, x, y, bounds=(0, [3e4, max(y)/max(x), 0.5])) 
        ax4.plot(x, exp_func(x, *(popt)), 'r-',
                label='$ ae^{bx}+c $ \na=%5.3f,\nb=%5.3f,\nc=%5.3f' % tuple(popt))
        ax4.set_ylabel("d50, мкм", fontdict={'size':12})
        ax4.set_xlabel(r'$ t,°C $', fontdict={'size':12})
        ax4.legend(loc='upper right', fontsize='large')
        ax4.tick_params(axis='both', which='major', labelsize=12)

        #пятый график
        ax5 = fig.add_subplot(3,2,5)
        df1= df.pivot_table(index='область_концентраций',columns='температура',values='насыпной_вес').loc[['высокая','средняя','малая','минимальная']]
        df1.index=[20,15,10,5]
        im = ax5.imshow(df1.values)
        ax5.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax5.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax5.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax5.set_xlabel(r'$ t,°C $', fontdict={'size':12})
        ax5.set_ylabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $', fontdict={'size':12})
        ax5.set_title('Насыпная плотность\nот температуры и концентрации', fontdict={'size':12})
        ax5.tick_params(axis='both', which='major', labelsize=12)

        #шестой график
        ax6 = fig.add_subplot(3,2,6)
        df1= df.pivot_table(index='область_концентраций',columns='температура',values='агрегаты_размер').loc[['высокая','средняя','малая','минимальная']]
        df1.index=[20,15,10,5]
        im = ax6.imshow(df1.values)
        ax6.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax6.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax6.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax6.set_xlabel(r'$ t,°C $')
        ax6.set_ylabel('$ C_{TiCl_{4}} \: на \: \omega TiO_{2},% $', fontdict={'size':12})
        ax6.set_title('Средний размер агрегатов\nот температуры и концентрации', fontdict={'size':12})
        ax6.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
        
        return fig
    
    def figure_4(us):
        
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        #без уз от t
        ax = fig.add_subplot(2,2,1)
        us1 = us.pivot(index='температура', columns='размер', values='без_уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']]
        im = ax.imshow(us1.values)
        ax.set_xticks(np.arange(len(us1.columns)), labels=us1.columns)
        ax.set_yticks(np.arange(len(us1.index)), labels=us1.index)

        for i in range(us1.values.shape[0]):
            for j in range(us1.values.shape[1]):
                text = ax.text(j, i, np.round(us1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax.set_ylabel(r'$ t,°C$', fontdict={'fontsize':10})
        ax.set_title('Доля агрегатов\nдо обработки УЗ',fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #уз от t
        ax2 = fig.add_subplot(2,2,3)
        us1 = us.pivot(index='температура', columns='размер', values='уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']]
        im = ax2.imshow(us1.values)
        ax2.set_xticks(np.arange(len(us1.columns)), labels=us1.columns)
        ax2.set_yticks(np.arange(len(us1.index)), labels=us1.index)

        for i in range(us1.values.shape[0]):
            for j in range(us1.values.shape[1]):
                text = ax2.text(j, i, np.round(us1.values[i, j], 2),
                            ha="center", va="center", color="w")

        ax2.set_ylabel(r'$ t,°C$', fontdict={'fontsize':10})
        ax2.set_xlabel('Фракция, мкм', fontdict={'fontsize':10})
        ax2.set_title('Доля агрегатов\nпосле УЗ', fontdict={'fontsize':10})
        ax2.tick_params(axis='both', which='major', labelsize=10)

        #гистограма до УЗ
        ax3 = fig.add_subplot(2,2,2)
        h120 = us.pivot(index='температура', columns='размер', values='без_уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']].loc[120]
        h160 = us.pivot(index='температура', columns='размер', values='без_уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']].loc[160]

        p1 = ax3.bar(h120.index, h120.values,width=0.5, linewidth=0)
        p2 = ax3.bar(h160.index, h160.values,width=0.3, linewidth=0)
        ax3.legend((p1[0], p2[0]), ('120°C', '160°C'), prop={'size':8})
        ax3.set_ylabel('процентная доля', fontdict={'fontsize':10})
        ax3.set_xlabel('Фракция, мкм', fontdict={'fontsize':10})
        ax3.tick_params(axis='both', which='major', labelsize=9)

        #гистограма после УЗ
        ax4 = fig.add_subplot(2,2,4)
        h120 = us.pivot(index='температура', columns='размер', values='уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']].loc[120]
        h160 = us.pivot(index='температура', columns='размер', values='уз')[['0-3', '3-6','6-9','9-30', '30-60','60-90', '90-300','300-600']].loc[160]

        p1 = ax4.bar(h120.index, h120.values,width=0.5, linewidth=0)
        p2 = ax4.bar(h160.index, h160.values,width=0.3, linewidth=0)
        ax4.legend((p1[0], p2[0]), ('120°C', '160°C'), prop={'size':8})
        ax4.set_ylabel('процентная доля', fontdict={'fontsize':10})
        ax4.set_xlabel('Фракция, мкм', fontdict={'fontsize':10})
        ax4.tick_params(axis='both', which='major', labelsize=9)
        
        return fig
    
    def figure_6(df):
        
        number_filter = df['Номер'] > 118
        t_filter = (df['температура'] > 120) & (df['температура'] < 145)
        c_filter = (df['концентрация'] > 15) & (df['концентрация'] < 16)
        time_filter = np.isnan(df['время_выдержки'])
        agent_type_filter = df['агент_тип'] == 'MgCl2'
        ratio_filter = (df['соотношение'] <= 0.2) & (df['соотношение'] >= 0.04)
        df_ = df[number_filter & t_filter & c_filter & time_filter & agent_type_filter & ratio_filter]

        fig = plt.figure(figsize=(14, 15))

        rows = 3
        columns = 12

        grid = plt.GridSpec(rows, columns, wspace = .4, hspace = .4)

        #125°C
        plt.subplot(grid[0, 0:4])

        anatase = df_.groupby(['температура','соотношение'])['анатаз'].mean().unstack('соотношение').loc[125]
        brukite = df_.groupby(['температура','соотношение'])['брукит'].mean().unstack('соотношение').loc[125]
        rutile = df_.groupby(['температура','соотношение'])['рутил'].mean().unstack('соотношение').loc[125]

        plt.scatter(anatase.index,anatase.values)
        anatase_ = np.polyval(np.polyfit(anatase.index,anatase.values,2), anatase.index)
        plt.plot(anatase.index,anatase_, label='анатаз')

        plt.scatter(brukite.index,brukite.values)
        brukite_ = np.polyval(np.polyfit(brukite.index,brukite.values,2), brukite.index)
        plt.plot(brukite.index,brukite_, label='брукит')

        plt.scatter(rutile.index,rutile.values)
        rutile_ = np.polyval(np.polyfit(rutile.index,rutile.values,2), rutile.index)
        plt.plot(rutile.index,rutile_, label='рутил')

        plt.legend(loc='best', fontsize='small')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_ylabel(r'$ Выход,~\% $', fontdict={'size':12})
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('125°C', fontdict={'size':12})

        #135°C
        plt.subplot(grid[0, 4:8])

        anatase = df_.groupby(['температура','соотношение'])['анатаз'].mean().unstack('соотношение').loc[135]
        brukite = df_.groupby(['температура','соотношение'])['брукит'].mean().unstack('соотношение').loc[135]
        rutile = df_.groupby(['температура','соотношение'])['рутил'].mean().unstack('соотношение').loc[135]

        plt.scatter(anatase.index,anatase.values)
        anatase_ = np.polyval(np.polyfit(anatase.index,anatase.values,2), anatase.index)
        plt.plot(anatase.index,anatase_, label='анатаз')

        plt.scatter(brukite.index,brukite.values)
        brukite_ = np.polyval(np.polyfit(brukite.index,brukite.values,2), brukite.index)
        plt.plot(brukite.index,brukite_, label='брукит')

        plt.scatter(rutile.index,rutile.values)
        rutile_ = np.polyval(np.polyfit(rutile.index,rutile.values,2), rutile.index)
        plt.plot(rutile.index,rutile_, label='рутил')

        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('135°C', fontdict={'size':12})

        #140°C
        plt.subplot(grid[0, 8:12])

        anatase = df_.groupby(['температура','соотношение'])['анатаз'].mean().unstack('соотношение').loc[140]
        brukite = df_.groupby(['температура','соотношение'])['брукит'].mean().unstack('соотношение').loc[140]
        rutile = df_.groupby(['температура','соотношение'])['рутил'].mean().unstack('соотношение').loc[140]

        plt.scatter(anatase.index,anatase.values)
        anatase_ = np.polyval(np.polyfit(anatase.index,anatase.values,2), anatase.index)
        plt.plot(anatase.index,anatase_, label='анатаз')

        plt.scatter(brukite.index,brukite.values)
        brukite_ = np.polyval(np.polyfit(brukite.index,brukite.values,2), brukite.index)
        plt.plot(brukite.index,brukite_, label='брукит')

        plt.scatter(rutile.index,rutile.values)
        rutile_ = np.polyval(np.polyfit(rutile.index,rutile.values,2), rutile.index)
        plt.plot(rutile.index,rutile_, label='рутил')

        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('140°C', fontdict={'size':12})

        #130°C
        plt.subplot(grid[1, :])
        df_ = df[number_filter & t_filter & c_filter & time_filter & agent_type_filter]

        anatase = df_.groupby(['температура','соотношение'])['анатаз'].mean().unstack('соотношение').loc[130]
        brukite = df_.groupby(['температура','соотношение'])['брукит'].mean().unstack('соотношение').loc[130]
        rutile = df_.groupby(['температура','соотношение'])['рутил'].mean().unstack('соотношение').loc[130]

        plt.scatter(anatase.index,anatase.values)
        anatase_ = np.polyval(np.polyfit(anatase.index,anatase.values,5), anatase.index)
        plt.plot(anatase.index,anatase_, label='анатаз')

        plt.scatter(brukite.index,brukite.values)
        brukite_ = np.polyval(np.polyfit(brukite.index,brukite.values,5), brukite.index)
        plt.plot(brukite.index,brukite_, label='брукит')

        plt.scatter(rutile.index,rutile.values)
        rutile_ = np.polyval(np.polyfit(rutile.index,rutile.values,5), rutile.index)
        plt.plot(rutile.index,rutile_, label='рутил')

        plt.legend(loc='best', fontsize='small')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_ylabel(r'$ Выход,~\% $', fontdict={'size':12})
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('130°C', fontdict={'size':12})

        df_ = df[number_filter & t_filter & c_filter & time_filter & agent_type_filter & ratio_filter]

        #Соотношение-температура анатаз
        plt.subplot(grid[2, 0:4])
        ax = plt.gca()
        df1 = df_.groupby(['температура','соотношение'])['анатаз'].mean().unstack('соотношение')
        im = ax.imshow(df1.values)
        ax.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax.set_ylabel(r'$ t,°C $', fontdict={'size':12})
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('$Анатаз$', fontdict={'size':12})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #Соотношение-температура брукит
        plt.subplot(grid[2, 4:8])
        ax = plt.gca()
        df1 = df_.groupby(['температура','соотношение'])['брукит'].mean().unstack('соотношение')
        im = ax.imshow(df1.values)
        ax.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax.set_ylabel(r'$ t,°C $', fontdict={'size':12})
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('$Брукит$', fontdict={'size':12})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #Соотношение-температура рутил
        plt.subplot(grid[2, 8:12])
        ax = plt.gca()
        df1 = df_.groupby(['температура','соотношение'])['рутил'].mean().unstack('соотношение')
        im = ax.imshow(df1.values)
        ax.set_xticks(np.arange(len(df1.columns)), labels=df1.columns)
        ax.set_yticks(np.arange(len(df1.index)), labels=df1.index)

        for i in range(df1.values.shape[0]):
            for j in range(df1.values.shape[1]):
                text = ax.text(j, i, np.round(df1.values[i, j], 2),
                            ha="center", va="center", color="w")
                
        ax.set_ylabel(r'$ t,°C $', fontdict={'size':12})
        ax.set_xlabel('$ \\frac{TiO_{2}}{MgCl_{2}} $')
        ax.set_title('$Рутил$', fontdict={'size':12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        return fig    
    
    def figure_10(df):
        
        def exp_f(x, a, b, c):
            return a * np.exp(b * x) + c
    
        ratio_filter = (df['соотношение'] <= 0.21) & (df['соотношение'] >= 0.2)
        concentration_filter = df['область_концентраций'] == 'малая'

        x = df[ratio_filter & concentration_filter].sort_values(by='температура')['температура'].values - 120
        anatase = df[ratio_filter & concentration_filter].sort_values(by='температура')['анатаз'].values / 100
        rutile = df[ratio_filter & concentration_filter].sort_values(by='температура')['рутил'].values / 100
        brukite = df[ratio_filter & concentration_filter].sort_values(by='температура')['брукит'].values /100

        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        #первый график
        ax = fig.add_subplot(2,2,1)
        param_bounds=([0.99,-1,0],[1,-0.1,0.01])
        popt, pcov = curve_fit(exp_f, x, rutile, bounds=param_bounds) 
        ax.plot(x, exp_f(x, *(popt)), 'g-',
                label='$ ae^{bx}+c $ \na=%5.3f,\nb=%5.3f,\nc=%5.3f' % tuple(popt))

        ax.scatter(x,anatase)
        ax.scatter(x,rutile)
        ax.scatter(x,brukite)

        ax.legend(loc='best', fontsize='small')
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ Относительная~температура,°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #второй график
        ax = fig.add_subplot(2,2,2)
        B_range = np.linspace(0.01,5,500)

        def model_f(x,A,B):
            rut = exp_f(x,1,A,0)
            br = exp_f(x,1-rut, A*B,0)
            an = 1 - rut - br
            return rut, br, an

        A = dfc.get_coeffs(x[0:4],np.log(rutile)[0:4],n=2)[0]
        B = B_range[B_range > cumulative_trapezoid(brukite, x)[-1]/cumulative_trapezoid(rutile, x)[-1]][0]
        B_ratio = np.array(cumulative_trapezoid(brukite,x)[-1]/cumulative_trapezoid(rutile,x)[-1])
        B_range = np.linspace(0.01,2.5,100)
        x_ = np.linspace(0,50,100)
        intergrals_ratio = np.array([(cumulative_trapezoid(model_f(x_,-1,B)[1],x_)[-1]
                            /cumulative_trapezoid(model_f(x_,-1,B)[0],x_)[-1])
                            for B in B_range])
        B = B_range[intergrals_ratio == intergrals_ratio[intergrals_ratio > B_ratio][-1]]

        ax.scatter(x,anatase)
        ax.scatter(x,rutile)
        ax.scatter(x,brukite)
        ax.plot(x,model_f(x,A,B)[1],'y-')
        ax.plot(x,model_f(x,A,B)[2])
        ax.text(0.7, 0.5, r'$ \frac{\gamma_{2}}{\gamma_{1}}$'+f'={np.round(B,2)}', fontdict={'size':12}, transform=ax.transAxes)
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ Относительная~температура,°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #третий график
        ax = fig.add_subplot(2,2,3)
        ax.plot(B_range, intergrals_ratio)
        ax.set_ylabel(r"$\frac{\int~f_{brukite}}{\int~f_{rutile}}$", fontdict={'fontsize':14})
        ax.set_xlabel(r'$ \frac{\gamma_{2}}{\gamma_{1}} $', fontdict={'fontsize':14})
        ax.tick_params(axis='both', which='major', labelsize=10)

        #четвертый график
        ax = fig.add_subplot(2,2,4)
        ax.plot(x_,model_f(x_,A,B)[0],'g-')
        ax.plot(x_,model_f(x_,A,B)[1],'y-')
        ax.plot(x_,model_f(x_,A,B)[2])
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ Относительная~температура,°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.show()
        df[ratio_filter & concentration_filter].sort_values(by='температура')
        
        return fig
    
    def figure_11(df):
        
        def exp_f(x, a, b, c):
            return a * np.exp(b * x) + c
        
        high_volume_filter = df['объем_реактора'] == 'стандартный'
        low_volume_filter = df['объем_реактора'] == 'малый'

        high_ratio_filter = (df['соотношение'] <= 0.22) & (df['соотношение'] >= 0.18)
        medium_ratio_filter = (df['соотношение'] <= 0.18) & (df['соотношение'] >= 0.13)
        low_ratio_filter = (df['соотношение'] <= 0.12) & (df['соотношение'] >= 0.09)

        ###
        fig = plt.figure(figsize=(16, 20))
        fig.subplots_adjust(wspace=0.3, hspace=0.2)

        ###
        df1 = df[high_ratio_filter & high_volume_filter].groupby(
            ['концентрация','температура'])[['соотношение','анатаз','рутил','брукит']].mean().unstack('температура').loc[15.1].dropna()

        ax = fig.add_subplot(4,3,1)

        x = df1['рутил'].index - 120
        y = df1['рутил'].values / 100
        ybr = df1['брукит'].values / 100
        yan = df1['анатаз'].values / 100

        ax.scatter(x,y)
        ax.scatter(x,ybr)
        ax.scatter(x,yan)
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'$С=15.3;~R=0.2;~V=150;$ ', fontdict={'fontsize':10})

        ###
        ax = fig.add_subplot(4,3,2)

        k = np.polyfit(x,np.log(y),1)[0]
        b = np.polyfit(x,np.log(y),1)[1]

        RMSE = np.array(
            [(np.square(np.log(y)[:q] - np.polyval(np.polyfit(x[:q],np.log(y)[:q],1),x[:q]))
            ).mean() for q in range(2,len(x)+1)])

        x_ = x[x <= x[1:][RMSE < (RMSE.std() + RMSE.mean())][-1]]

        ax.scatter(x,np.log(y))
        ax.plot(x_,dfc.get_fit(x_,np.log(y)[0:len(x_)],n=2))
        ax.set_ylabel(r'$\ln(C_{rutile})$', fontdict={'size':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'size':12})
        ax.set_title('$ Поиск~А~и~C_{rutile}=0.01 $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.75, 
                '$ kx~+~b $\n' 
                +f'k={np.round(k,2)}\n'
                +f'b={np.round(b,2)}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ###
        ax = fig.add_subplot(4,3,3)

        x_ = np.append(x,(np.log(0.01)-b)/k)
        y_ = np.append(y,0.01)
        ybr_ = np.append(ybr,0.01)
        yan_ = np.append(yan,0.99)
        ax.scatter(x_,y_)
        ax.scatter(x_,ybr_)
        ax.scatter(x_,yan_)

        def model_f(x,A,B):
            rut = exp_f(x,1,A,0)
            br = exp_f(x,1-rut, A*B,0)
            an = 1 - rut - br
            return rut, br, an

        B_range = np.linspace(0.01,2.5,100)
        _x = np.linspace(min(x_),max(x_),100)
        intergrals_ratio = np.array([(cumulative_trapezoid(model_f(_x,k,B)[1],_x)[-1]
                            /cumulative_trapezoid(model_f(_x,k,B)[0],_x)[-1])
                            for B in B_range])
        B_ratio = np.array(cumulative_trapezoid(ybr_,x_)[-1]/cumulative_trapezoid(y_,x_)[-1])
        B = B_range[intergrals_ratio == intergrals_ratio[intergrals_ratio > B_ratio][-1]]


        ax.plot(_x, model_f(_x,k,B)[0])
        ax.plot(_x, model_f(_x,k,B)[1])
        ax.plot(_x, model_f(_x,k,B)[2])
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.set_title('$ Результат~моделирования $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.5, 
                r'$ A $' 
                +f'={round(max(x_)/50,2)}\n'
                +r'$B$'
                +f'={str(*np.round(B,2))}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ##############################################################################################################################
        df1 = df[medium_ratio_filter & high_volume_filter].groupby(
            ['концентрация','температура'])[['соотношение','анатаз','рутил','брукит']].mean().unstack('температура').loc[15.3]

        ax = fig.add_subplot(4,3,4)

        x = df1['рутил'].index - 125
        y = df1['рутил'].values / 100
        ybr = df1['брукит'].values / 100
        yan = df1['анатаз'].values / 100

        ax.scatter(x,y)
        ax.scatter(x,ybr)
        ax.scatter(x,yan)
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'$С=15.3;~R=0.15;~V=150;$ ', fontdict={'fontsize':10})
        ###
        ax = fig.add_subplot(4,3,5)

        k = np.polyfit(x,np.log(y),1)[0]
        b = np.polyfit(x,np.log(y),1)[1]

        RMSE = np.array(
            [(np.square(np.log(y)[:q] - np.polyval(np.polyfit(x[:q],np.log(y)[:q],1),x[:q]))
            ).mean() for q in range(2,len(x)+1)])

        x_ = x[x <= x[1:][RMSE < (RMSE.std() + RMSE.mean())][-1]]

        ax.scatter(x,np.log(y))
        ax.plot(x_,dfc.get_fit(x_,np.log(y)[0:len(x_)],n=2))
        ax.set_ylabel(r'$\ln(C_{rutile})$', fontdict={'size':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'size':12})
        ax.set_title('$ Поиск~А~и~C_{rutile}=0.01 $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.75, 
                '$ kx~+~b $\n' 
                +f'k={np.round(k,2)}\n'
                +f'b={np.round(b,2)}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ###
        ax = fig.add_subplot(4,3,6)

        x_ = np.append(x,(np.log(0.01)-b)/k)
        y_ = np.append(y,0.01)
        ybr_ = np.append(ybr,0.01)
        yan_ = np.append(yan,0.99)
        ax.scatter(x_,y_)
        ax.scatter(x_,ybr_)
        ax.scatter(x_,yan_)

        def model_f(x,A,B):
            rut = exp_f(x,1,A,0)
            br = exp_f(x,1-rut, A*B,0)
            an = 1 - rut - br
            return rut, br, an

        B_range = np.linspace(0.01,2.5,100)
        _x = np.linspace(min(x_),max(x_),100)
        intergrals_ratio = np.array([(cumulative_trapezoid(model_f(_x,k,B)[1],_x)[-1]
                            /cumulative_trapezoid(model_f(_x,k,B)[0],_x)[-1])
                            for B in B_range])
        B_ratio = np.array(cumulative_trapezoid(ybr_,x_)[-1]/cumulative_trapezoid(y_,x_)[-1])
        B = B_range[intergrals_ratio == intergrals_ratio[intergrals_ratio > B_ratio][-1]]

        ax.plot(_x, model_f(_x,k,B)[0])
        ax.plot(_x, model_f(_x,k,B)[1])
        ax.plot(_x, model_f(_x,k,B)[2])
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.set_title('$ Результат~моделирования $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.5, 
                r'$ A $' 
                +f'={round(max(x_)/50,2)}\n'
                +r'$B$'
                +f'={str(*np.round(B,2))}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ####################################################################################################################################
        df1 = df[low_ratio_filter & high_volume_filter].groupby(
            ['концентрация','температура'])[['соотношение','анатаз','рутил','брукит']].mean().unstack('температура').loc[15.1].dropna()

        ax = fig.add_subplot(4,3,7)

        x = df1['рутил'].drop(120).index - 125
        y = df1['рутил'].drop(120).values / 100
        ybr = df1['брукит'].drop(120).values / 100
        yan = df1['анатаз'].drop(120).values / 100

        ax.scatter(x,y)
        ax.scatter(x,ybr)
        ax.scatter(x,yan)
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'$С=15.1;~R=0.1;~V=150;$ ', fontdict={'fontsize':10})
        ###
        ax = fig.add_subplot(4,3,8)

        k = np.polyfit(x,np.log(y),1)[0]
        b = np.polyfit(x,np.log(y),1)[1]

        x_ = x

        ax.scatter(x,np.log(y))
        ax.plot(x_,dfc.get_fit(x_,np.log(y)[0:len(x_)],n=2))
        ax.set_ylabel(r'$\ln(C_{rutile})$', fontdict={'size':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'size':12})
        ax.set_title('$ Поиск~А~и~C_{rutile}=0.01 $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.75, 
                '$ kx~+~b $\n' 
                +f'k={np.round(k,2)}\n'
                +f'b={np.round(b,2)}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ###
        ax = fig.add_subplot(4,3,9)

        if y[-1] > 0.05:
            x_ = np.append(x,(np.log(0.01)-b)/k)
            y_ = np.append(y,0.01)
            ybr_ = np.append(ybr,0.01)
            yan_ = np.append(yan,0.99)
        else:
            x_ = x
            y_ = y
            ybr_ = ybr
            yan_ = yan

        ax.scatter(x_,y_)
        ax.scatter(x_,ybr_)
        ax.scatter(x_,yan_)

        def model_f(x,A,B):
            rut = exp_f(x,1,A,0)
            br = exp_f(x,1-rut, A*B,0)
            an = 1 - rut - br
            return rut, br, an

        B_range = np.linspace(0.01,2.5,100)
        _x = np.linspace(min(x_),max(x_),100)
        intergrals_ratio = np.array([(cumulative_trapezoid(model_f(_x,k,B)[1],_x)[-1]
                            /cumulative_trapezoid(model_f(_x,k,B)[0],_x)[-1])
                            for B in B_range])
        B_ratio = np.array(cumulative_trapezoid(ybr_,x_)[-1]/cumulative_trapezoid(y_,x_)[-1])
        B = B_range[intergrals_ratio == intergrals_ratio[intergrals_ratio > B_ratio][-1]]

        ax.plot(_x, model_f(_x,k,B)[0])
        ax.plot(_x, model_f(_x,k,B)[1])
        ax.plot(_x, model_f(_x,k,B)[2])
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.set_title('$ Результат~моделирования $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.4, 
                r'$ A $' 
                +f'={round(max(x_)/50,2)}\n'
                +r'$B$'
                +f'={str(*np.round(B,2))}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ###############################################################################################################################
        df1 = df[low_ratio_filter & low_volume_filter].groupby(
            ['концентрация','температура'])[['соотношение','анатаз','рутил','брукит']].mean().unstack('температура').loc[11.5].dropna()

        ax = fig.add_subplot(4,3,10)

        x = df1['рутил'].drop(160).index - 120
        y = df1['рутил'].drop(160).values / 100
        ybr = df1['брукит'].drop(160).values / 100
        yan = df1['анатаз'].drop(160).values / 100

        ax.scatter(x,y)
        ax.scatter(x,ybr)
        ax.scatter(x,yan)
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'$С=11.5;~R=0.1;~V=80;$ ', fontdict={'fontsize':10})
        ###
        ax = fig.add_subplot(4,3,11)

        k = np.polyfit(x[:2],np.log(y[:2]),1)[0]
        b = np.polyfit(x[:2],np.log(y[:2]),1)[1]

        ax.scatter(x[:2],np.log(y[:2]))
        ax.plot(x_,np.polyval(np.polyfit(x[:2],np.log(y[:2]),1),x_))
        ax.set_ylabel(r'$\ln(C_{rutile})$', fontdict={'size':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'size':12})
        ax.set_title('$ Поиск~А~и~C_{rutile}=0.01 $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.75, 
                '$ kx~+~b $\n' 
                +f'k={np.round(k,2)}\n'
                +f'b={np.round(b,2)}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ###
        ax = fig.add_subplot(4,3,12)

        if y[-1] > 0.05:
            x_ = np.append(x,(np.log(0.01)-b)/k)
            y_ = np.append(y,0.01)
            ybr_ = np.append(ybr,0.01)
            yan_ = np.append(yan,0.99)
        else:
            x_ = x
            y_ = y
            ybr_ = ybr
            yan_ = yan

        ax.scatter(x_,y_)
        ax.scatter(x_,ybr_)
        ax.scatter(x_,yan_)

        def model_f(x,A,B):
            rut = exp_f(x,1,A,0)
            br = exp_f(x,1-rut, A*B,0)
            an = 1 - rut - br
            return rut, br, an

        B_range = np.linspace(0.01,2.5,100)
        _x = np.linspace(min(x_),max(x_),100)
        intergrals_ratio = np.array([(cumulative_trapezoid(model_f(_x,k,B)[1],_x)[-1]
                            /cumulative_trapezoid(model_f(_x,k,B)[0],_x)[-1])
                            for B in B_range])
        B_ratio = np.array(cumulative_trapezoid(ybr_,x_)[-1]/cumulative_trapezoid(y_,x_)[-1])
        B = B_range[intergrals_ratio == intergrals_ratio[intergrals_ratio > B_ratio][-1]]

        ax.plot(_x, model_f(_x,k,B)[0])
        ax.plot(_x, model_f(_x,k,B)[1])
        ax.plot(_x, model_f(_x,k,B)[2])
        ax.set_ylabel("Выход", fontdict={'fontsize':12})
        ax.set_xlabel(r'$ t_{отн},°C $', fontdict={'fontsize':12})
        ax.set_title('$ Результат~моделирования $', fontdict={'fontsize':10})
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.text(0.75, 0.4, 
                r'$ A $' 
                +f'={round(max(x_)/50,2)}\n'
                +r'$B$'
                +f'={str(*np.round(B,2))}',
                fontdict={'size':12}, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))

        plt.show()
        return fig