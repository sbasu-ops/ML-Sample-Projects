#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[49]:


file = pd.read_csv('cse6242_us_stock_db.csv')
display(file)
df = file


# In[47]:


file.dtypes


# In[54]:


def cleandata(df):
    #rename column1 'date'
    df = df.rename(columns={'Unnamed: 0': 'date'})
    
    #rename column2 'tick and country'
    df = df.rename(columns={'Unnamed: 1': 'tick and country'})
    
    #drop '().value' from column names
    df = df.rename(columns = lambda x: x[:-8] if x.endswith('().value') else x)
    
    #fill #N/As
    values = {'id_isin': '', #set all #N/As in string type columns to ''
              'id_cusip': '',
              'primary_exchange_name': '',
              'gics_sector_name': '',
              'gics_industry_name': '',
              'gics_sub_industry_name': '',
              'px_last': 0, #set #N/As to 0
              'rsi': 50, #set #N/As to 50, neither overbought nor oversold
              'macd': 0, #set #N/As to 0
              'boll': -1, #set #N/As to -1, nonsensical answer to indicate data isn't available
              'px_volume': 0, #set #N/As to 0
              'eqy_sh_out': 0, 
              'pe_ratio': 0,
              'dividend_yield': 0,
              'px_to_book_ratio': 0,
              'px_to_sales_ratio': 0,
              'ev_to_ebitda': 0,
              'curr_entp_val': 0,
              'eps_growth': 0,
              'basic_eps_5yr_avg_gr': 0,
              'growth_in_cap': 0,
              'bvps_growth': 0,
              'rd_expend_yr_growth': 0,
              'retention_ratio': 0,
              'net_rev_growth': 0,
              'empl_growth': 0,
              'asset_growth': 0,
              'is_div_per_shr': 0,
              'bok_val_per_sh': 0,
              'cf_free_cash_flow': 0,
              'free_cash_flow_per_sh_growh': 0,
              'ebitda': 0,
              'ebit': 0,
              'oper_margin': 0,
              'pretax_margin': 0,
              'return_com_eqy': 0,
              'return_on_asset': 0,
              'cb_is_roic': 0,
              'return_on_cap': 0,
              'asset_turnover': 0,
              'px_to_cash_flow': 0,
              'cf_net_inc': 0,
              'dvd_payout_ratio': 0,
              'cash_gen_to_cash_req': 0,
              'cash_dvd_coverage': 0,
              'cfo_to_sales': 0,
              'cur_ratio': 0,
              'quick_ratio': 0,
              'tot_debt_to_tot_asset': 0,
              'tot_debt_to_com_eqy': 0,
              'acct_rev_turn': 0,
              'invent_turn': 0,
              'gross_margin': 0,
              'ebitda_to_tot_int_expn': 0,
              'sharpe_ratio': 1, #set #N/As to 1 return:risk ratio of stock is "balanced", >1 excess returns to risk
              'beta': 1, #set #N/A to 1 assume correlated with market
             } 
    df = df.fillna(value=values)
    
    #drop rows with no primary exchange, ticker symbol, last trade price or shares outstanding as they're not readily tradeable
    df = df[df.primary_exchange_name != '']
    df = df[df.security_des != '']
    df = df[df.px_last != 0]
    df = df[df.eqy_sh_out != 0]
    
    #drop columns cb_is_roic and ebitda_to_tot_int_expn as they almost no data
    df = df.drop(['cb_is_roic', 'ebitda_to_tot_int_expn'], axis=1)
    
    return df
    


# In[55]:


df = cleandata(df)
display(df)


# In[ ]:




