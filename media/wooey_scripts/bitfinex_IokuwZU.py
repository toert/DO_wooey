from collections import OrderedDict
import sys
import requests  # pip install requests
import json
import base64
import hashlib
import hmac
import os
import tensorflow as tf
import numpy as np
import time  # for nonce


ROUND_DIGITS = 5


def norm_price_batch(batch):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.divide(batch, batch[:,-1, np.newaxis])
        c[ ~ np.isfinite( c )] = 1  # -inf inf NaN
    return c.reshape(1, c.shape[0], c.shape[1], 1)

# функция возвращает рекомендованный портфель
def calculate_recommended_proportions(M, model_name):
    tf.reset_default_graph()
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], model_name)
    portfolio_op = sess.graph.get_tensor_by_name("portfolio/Softmax:0")
    M_norm = norm_price_batch(M)
    predictions = sess.run(portfolio_op, {'X:0': M_norm})
    return predictions[0]



class BitfinexClient(object):

    def __init__(self, key, secret, base='https://api.bitfinex.com/'):
        self.BASE_URL = base
        self.KEY = key
        self.SECRET = secret

    def _nonce(self):
        """
        Returns a nonce
        Used in authentication
        """
        return str(int(round(time.time() * 10000)))

    def _headers(self, path, nonce, body):
        signature = "/api/" + path + nonce + body
        h = hmac.new(self.SECRET.encode('ascii'), signature.encode('ascii'), hashlib.sha384)
        signature = h.hexdigest()
        return {
            "bfx-nonce": nonce,
            "bfx-apikey": self.KEY,
            "bfx-signature": signature,
            "content-type": "application/json"
        }

    def _jsonify_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            print(response.status_code)
            print(response.text)

    def _sign_payload(self, payload):
        j = json.dumps(payload)
        data = base64.standard_b64encode(j.encode('utf8'))

        h = hmac.new(self.SECRET.encode('utf8'), data, hashlib.sha384)
        signature = h.hexdigest()
        return {
            "X-BFX-APIKEY": self.KEY,
            "X-BFX-SIGNATURE": signature,
            "X-BFX-PAYLOAD": data
        }

    def new_order(self, symbol, amount, price, side, ord_type, exchange='bitfinex', **kwargs):
        payload = {
            "request": "/v1/order/new",
            "nonce": self._nonce(),
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "exchange": exchange,
            "side": side,
            "type": ord_type,
        }
        print(payload)
        payload = dict(**payload, **kwargs)
        signed_payload = self._sign_payload(payload)
        response = requests.post(self.BASE_URL + "v1/order/new", headers=signed_payload, verify=True)
        return self._jsonify_response(response)

    def get_wallet(self):
        signed_payload = self._sign_payload({"request": '/v1/balances',
                                             'nonce': self._nonce()})
        response = requests.post(self.BASE_URL + "v1/balances", headers=signed_payload, verify=True)
        return self._jsonify_response(response)

    def req(self, path, params={}):
        nonce = self._nonce()
        body = params
        rawBody = json.dumps(body)
        headers = self._headers(path, nonce, rawBody)
        url = self.BASE_URL + path
        resp = requests.post(url, headers=headers, data=rawBody, verify=True)
        return resp

    def get_orders(self):
        response = self.req('v2/auth/r/orders')
        return self._jsonify_response(response)

    def get_amounts_from_wallets(self, wallets, symbols):
        prices = {}
        wallets = self.get_wallet()
        wallets = [
            {
              "type":"deposit",
              "currency":"btc",
              "amount":"0.0",
              "available":"0.0"
            },{
              "type": "exchange",
              "currency": "ltc",
              "amount": "112.0",
              "available": "1.0"
            },{
              "type": "exchange",
              "currency": "btc",
              "amount": "12",
              "available": "1"
            },{
              "type": "exchange",
              "currency": "eth",
              "amount": "12",
              "available": "1"
            },{
              "type":"trading",
              "currency":"btc",
              "amount":"1",
              "available":"1"
            },{
              "type":"trading",
              "currency":"usd",
              "amount":"1",
              "available":"1"
        }]
        if not wallets:
            return [0 for _ in prices]
        for symbol in symbols:
            wallet = list(filter(lambda wal: wal['currency'] == symbol and wal['type'] == 'exchange', wallets))
            if wallet:
                prices[symbol] = float(wallet[0]['amount'])
            else:
                prices[symbol] = 0
        return prices

    def set_new_amounts(self, current_amounts, new_amounts, main_symbol='btc'):
        deltas = {}
        for symbol, amount in current_amounts.items():
            deltas[symbol] = round(new_amounts[symbol]-amount, ROUND_DIGITS)
        # deltas = OrderedDict(sorted(deltas.items(), key=lambda x: x[1]))
        deltas[main_symbol] = 0
        for symbol, amount in ((sym, amount) for (sym, amount) in deltas.items() if amount<0):
            self.new_order(symbol=symbol+main_symbol, amount=-amount-(1/10**ROUND_DIGITS), price=9999999,
                           side='sell', ord_type='market', is_hidden=True)
        while self.get_orders():
            if time.time()-start_time > T*60:
                sys.exit('Не сумел продать валюту за время=T')
            time.sleep(10)
        for symbol, amount in ((sym, amount) for (sym, amount) in deltas.items() if amount > 0):
            self.new_order(symbol=symbol+main_symbol, amount=str(amount), price=0.001,
                           side='buy', ord_type='market', is_hidden=True)

    def get_candles(self, pair, timeframe=None, section='hist', **kwargs):
        valid_tfs = ['1m', '5m', '15m', '30m', '1h', '3h', '6h', '12h', '1D',
                     '7D', '14D', '1M']
        if timeframe:
            if type(timeframe) is int:
                timeframe = '{}m'.format(timeframe)
            if timeframe not in valid_tfs:
                raise ValueError("timeframe must be any of %s" % valid_tfs)
        else:
            timeframe = '30m'
        pair = 't' + pair if not pair.startswith('t') else pair
        key = 'trade:{}:{}/{}'.format(timeframe, pair, section)
        url = '{0}v2/candles/{1}/'.format(self.BASE_URL, key)
        response = requests.get(url, params=kwargs)
        print(response.url)
        return self._jsonify_response(response)


if __name__ == '__main__':
    client = BitfinexClient('ZV2WajVkRX13Whz0pAi2P26IZqeDjvt7TpGuX3ea9rc',
                            '6LLSkY4BvKPGudKJMx56D3mxZ17iL3Wp3unilA9pdhy')
    #m = 7
    main_symbol = 'btc'
    symbols = ['omg', 'ltc', 'eth', 'xmr', 'eos', 'bcc']
    all_symbols = symbols[:]
    all_symbols.append(main_symbol)
    print(all_symbols)
    n = 50
    T = 30
    P = 100
    M = []
    prices = {}
    wallets = client.get_wallet()

    while True:
        start_time = time.time()
        for symbol in symbols:
            pair_symbol = symbol + main_symbol
            print(pair_symbol)
            candles = client.get_candles(pair_symbol.upper(), timeframe=T, limit=n)
            prices[symbol] = candles[0][3]
            print(len([frame[3] for frame in reversed(candles)]))
            M.append([frame[3] for frame in reversed(candles)])
        M.append([1] * n)
        prices[main_symbol] = 1
        print(prices)
        M = np.array(M)
        print(M)
        recommended_proportions = calculate_recommended_proportions(M, "test_build")

        print(recommended_proportions)

        # recommended_proportions = [0.37, 0.1, 0.13, 0.4]
        current_amounts = client.get_amounts_from_wallets(wallets, all_symbols) # returns all wallets in BTC
        print(current_amounts)
        total_amount = sum([amount * prices[symbol] for symbol, amount in current_amounts.items()])
        recommended_amounts_btc = [total_amount*proportion for proportion in recommended_proportions]
        print(recommended_amounts_btc)
        recommended_amounts = {}
        for num, price in enumerate(recommended_proportions):
            symbol = all_symbols[num]
            recommended_amounts[symbol] = recommended_amounts_btc[num] / prices[symbol]
        print('Recommended amounts:', recommended_amounts)


        client.set_new_amounts(current_amounts, recommended_amounts)
        print(symbols)
        print('Сплю еще ',T*60 - (time.time() - start_time), 'секунд')
        time.sleep(T*60 - (time.time() - start_time))
        #print(client.new_order('0.00001', '333', 'buy', 'market', wallets[0]))
        #print(client.get_candles('BTCUSD'))
        #print(client.get_wallet())