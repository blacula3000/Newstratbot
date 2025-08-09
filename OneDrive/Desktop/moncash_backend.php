<?php
/**
 * MonCash Backend Integration for Lyly's Restaurant
 * This PHP file handles server-side MonCash payment processing
 */

class MonCashAPI {
    private $clientId;
    private $clientSecret;  
    private $businessKey;
    private $sandbox;
    private $baseUrl;
    
    public function __construct($config) {
        $this->clientId = $config['clientId'];
        $this->clientSecret = $config['clientSecret'];
        $this->businessKey = $config['businessKey'];
        $this->sandbox = $config['sandbox'] ?? true;
        $this->baseUrl = $this->sandbox 
            ? 'https://sandbox.moncashbutton.digicelgroup.com'
            : 'https://moncashbutton.digicelgroup.com';
    }
    
    /**
     * Get access token from MonCash API
     */
    public function getAccessToken() {
        $curl = curl_init();
        
        curl_setopt_array($curl, array(
            CURLOPT_URL => $this->baseUrl . '/Api/oauth/token',
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_ENCODING => '',
            CURLOPT_MAXREDIRS => 10,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
            CURLOPT_CUSTOMREQUEST => 'POST',
            CURLOPT_POSTFIELDS => http_build_query([
                'grant_type' => 'client_credentials',
                'client_id' => $this->clientId,
                'client_secret' => $this->clientSecret
            ]),
            CURLOPT_HTTPHEADER => array(
                'Content-Type: application/x-www-form-urlencoded'
            ),
        ));
        
        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        
        curl_close($curl);
        
        if ($httpCode !== 200) {
            throw new Exception('Failed to get access token: ' . $response);
        }
        
        $data = json_decode($response, true);
        return $data['access_token'];
    }
    
    /**
     * Create a payment request
     */
    public function createPayment($orderData) {
        $accessToken = $this->getAccessToken();
        
        $paymentData = [
            'amount' => $orderData['total'],
            'orderId' => $this->generateOrderId(),
        ];
        
        $curl = curl_init();
        
        curl_setopt_array($curl, array(
            CURLOPT_URL => $this->baseUrl . '/Api/v1/CreatePayment',
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_ENCODING => '',
            CURLOPT_MAXREDIRS => 10,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
            CURLOPT_CUSTOMREQUEST => 'POST',
            CURLOPT_POSTFIELDS => json_encode($paymentData),
            CURLOPT_HTTPHEADER => array(
                'Authorization: Bearer ' . $accessToken,
                'Content-Type: application/json'
            ),
        ));
        
        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        
        curl_close($curl);
        
        if ($httpCode !== 200) {
            throw new Exception('Failed to create payment: ' . $response);
        }
        
        return json_decode($response, true);
    }
    
    /**
     * Verify payment status
     */
    public function verifyPayment($transactionId) {
        $accessToken = $this->getAccessToken();
        
        $curl = curl_init();
        
        curl_setopt_array($curl, array(
            CURLOPT_URL => $this->baseUrl . '/Api/v1/RetrieveTransactionPayment',
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_ENCODING => '',
            CURLOPT_MAXREDIRS => 10,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
            CURLOPT_CUSTOMREQUEST => 'POST',
            CURLOPT_POSTFIELDS => json_encode([
                'transactionId' => $transactionId
            ]),
            CURLOPT_HTTPHEADER => array(
                'Authorization: Bearer ' . $accessToken,
                'Content-Type: application/json'
            ),
        ));
        
        $response = curl_exec($curl);
        $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
        
        curl_close($curl);
        
        if ($httpCode !== 200) {
            throw new Exception('Failed to verify payment: ' . $response);
        }
        
        return json_decode($response, true);
    }
    
    /**
     * Generate unique order ID
     */
    private function generateOrderId() {
        return 'LYLY-' . time() . '-' . substr(md5(uniqid()), 0, 8);
    }
}

/**
 * Order Management Class
 */
class OrderManager {
    private $dbFile;
    
    public function __construct($dbFile = 'orders.json') {
        $this->dbFile = $dbFile;
    }
    
    /**
     * Save order to file/database
     */
    public function saveOrder($orderData) {
        $orders = $this->getOrders();
        $orderData['id'] = uniqid();
        $orderData['created_at'] = date('Y-m-d H:i:s');
        $orderData['status'] = 'pending';
        
        $orders[] = $orderData;
        
        file_put_contents($this->dbFile, json_encode($orders, JSON_PRETTY_PRINT));
        
        return $orderData['id'];
    }
    
    /**
     * Update order status
     */
    public function updateOrderStatus($orderId, $status, $transactionId = null) {
        $orders = $this->getOrders();
        
        foreach ($orders as &$order) {
            if ($order['id'] === $orderId) {
                $order['status'] = $status;
                $order['updated_at'] = date('Y-m-d H:i:s');
                if ($transactionId) {
                    $order['transaction_id'] = $transactionId;
                }
                break;
            }
        }
        
        file_put_contents($this->dbFile, json_encode($orders, JSON_PRETTY_PRINT));
    }
    
    /**
     * Get all orders
     */
    private function getOrders() {
        if (!file_exists($this->dbFile)) {
            return [];
        }
        
        $content = file_get_contents($this->dbFile);
        return json_decode($content, true) ?: [];
    }
    
    /**
     * Send order notification
     */
    public function sendOrderNotification($orderData, $isPaid = false) {
        $status = $isPaid ? 'PAYÉE via MonCash' : 'À PAYER à la livraison';
        $emoji = $isPaid ? '💳✅' : '💰📦';
        
        $message = "$emoji NOUVELLE COMMANDE - $status\n\n";
        
        if ($isPaid && isset($orderData['transaction_id'])) {
            $message .= "📧 Référence MonCash: {$orderData['transaction_id']}\n\n";
        }
        
        $message .= "👤 Client: {$orderData['customerName']}\n";
        $message .= "📱 Téléphone: {$orderData['customerPhone']}\n";
        $message .= "📍 Adresse: {$orderData['deliveryAddress']}\n\n";
        
        $message .= "🍽️ Commande:\n";
        foreach ($orderData['items'] as $item) {
            $option = $item['selectedOption'] ? " ({$item['selectedOption']})" : '';
            $total = number_format($item['price'] * $item['quantity']);
            $message .= "• {$item['name']}{$option} x{$item['quantity']} = {$total} Gds\n";
        }
        
        $totalAmount = number_format($orderData['total']);
        $message .= "\n💰 Total: {$totalAmount} Gds";
        
        if ($isPaid) {
            $message .= " (DÉJÀ PAYÉ)";
        }
        
        if (!empty($orderData['specialInstructions'])) {
            $message .= "\n📝 Instructions: {$orderData['specialInstructions']}";
        }
        
        $message .= "\n⏰ Reçu: " . date('d/m/Y à H:i');
        
        // Send to WhatsApp (you can also integrate with email or SMS)
        $phone = '50937525134'; // Restaurant phone number
        $whatsappUrl = "https://wa.me/{$phone}?text=" . urlencode($message);
        
        return $whatsappUrl;
    }
}

// Handle API requests
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    
    // CORS headers
    header('Access-Control-Allow-Origin: *');
    header('Access-Control-Allow-Methods: POST, GET, OPTIONS');
    header('Access-Control-Allow-Headers: Content-Type');
    
    $input = json_decode(file_get_contents('php://input'), true);
    $action = $_GET['action'] ?? '';
    
    // MonCash configuration (store these securely, preferably in environment variables)
    $moncashConfig = [
        'clientId' => $_ENV['MONCASH_CLIENT_ID'] ?? 'YOUR_CLIENT_ID',
        'clientSecret' => $_ENV['MONCASH_CLIENT_SECRET'] ?? 'YOUR_CLIENT_SECRET',
        'businessKey' => $_ENV['MONCASH_BUSINESS_KEY'] ?? 'YOUR_BUSINESS_KEY',
        'sandbox' => true // Set to false for production
    ];
    
    $monCash = new MonCashAPI($moncashConfig);
    $orderManager = new OrderManager();
    
    try {
        switch ($action) {
            case 'create_payment':
                // Save order first
                $orderId = $orderManager->saveOrder($input);
                
                // Create MonCash payment
                $paymentResult = $monCash->createPayment($input);
                $paymentResult['order_id'] = $orderId;
                
                echo json_encode([
                    'success' => true,
                    'data' => $paymentResult
                ]);
                break;
                
            case 'verify_payment':
                $transactionId = $input['transactionId'];
                $orderId = $input['orderId'] ?? '';
                
                $paymentStatus = $monCash->verifyPayment($transactionId);
                
                if ($paymentStatus['status'] === 'SUCCESS') {
                    // Update order status
                    if ($orderId) {
                        $orderManager->updateOrderStatus($orderId, 'paid', $transactionId);
                    }
                    
                    // Send notification to restaurant
                    $orderData = $input['orderData'] ?? [];
                    $orderData['transaction_id'] = $transactionId;
                    $notificationUrl = $orderManager->sendOrderNotification($orderData, true);
                    
                    echo json_encode([
                        'success' => true,
                        'status' => 'paid',
                        'notification_url' => $notificationUrl
                    ]);
                } else {
                    echo json_encode([
                        'success' => false,
                        'status' => 'failed',
                        'message' => 'Payment verification failed'
                    ]);
                }
                break;
                
            case 'process_cash_order':
                // Save cash order
                $orderId = $orderManager->saveOrder($input);
                
                // Send notification
                $notificationUrl = $orderManager->sendOrderNotification($input, false);
                
                echo json_encode([
                    'success' => true,
                    'order_id' => $orderId,
                    'notification_url' => $notificationUrl
                ]);
                break;
                
            default:
                throw new Exception('Invalid action');
        }
        
    } catch (Exception $e) {
        http_response_code(400);
        echo json_encode([
            'success' => false,
            'error' => $e->getMessage()
        ]);
    }
} else {
    // Show basic info
    echo json_encode([
        'service' => 'Lyly\'s Restaurant MonCash API',
        'version' => '1.0',
        'endpoints' => [
            'POST ?action=create_payment',
            'POST ?action=verify_payment', 
            'POST ?action=process_cash_order'
        ]
    ]);
}
?>