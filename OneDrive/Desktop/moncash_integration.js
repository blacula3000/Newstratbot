// MonCash Payment Integration for Lyly's Restaurant
// This file handles MonCash payment processing

class MonCashPayment {
    constructor(config) {
        this.clientId = config.clientId;
        this.clientSecret = config.clientSecret;
        this.businessKey = config.businessKey;
        this.sandbox = config.sandbox || true;
        this.baseUrl = this.sandbox 
            ? 'https://sandbox.moncashbutton.digicelgroup.com'
            : 'https://moncashbutton.digicelgroup.com';
    }

    // Get access token from MonCash API
    async getAccessToken() {
        try {
            const response = await fetch(`${this.baseUrl}/Api/oauth/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'grant_type': 'client_credentials',
                    'client_id': this.clientId,
                    'client_secret': this.clientSecret
                })
            });

            const data = await response.json();
            return data.access_token;
        } catch (error) {
            console.error('Error getting access token:', error);
            throw error;
        }
    }

    // Create payment order
    async createPayment(orderData) {
        try {
            const accessToken = await this.getAccessToken();
            
            const paymentRequest = {
                amount: orderData.total,
                orderId: this.generateOrderId(),
                // MonCash requires specific format
            };

            const response = await fetch(`${this.baseUrl}/Api/v1/CreatePayment`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(paymentRequest)
            });

            const paymentData = await response.json();
            
            // Redirect user to MonCash payment page
            window.location.href = paymentData.payment_url;
            
            return paymentData;
        } catch (error) {
            console.error('Error creating payment:', error);
            throw error;
        }
    }

    // Verify payment status
    async verifyPayment(transactionId) {
        try {
            const accessToken = await this.getAccessToken();
            
            const response = await fetch(`${this.baseUrl}/Api/v1/RetrieveTransactionPayment`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transactionId: transactionId
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Error verifying payment:', error);
            throw error;
        }
    }

    // Generate unique order ID
    generateOrderId() {
        return `LYLY-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
}

// Enhanced order processing with MonCash integration
function enhancedProcessMoncashPayment(orderData) {
    // MonCash configuration (these should be stored securely on the server)
    const moncashConfig = {
        clientId: 'YOUR_CLIENT_ID', // Replace with actual Client ID
        clientSecret: 'YOUR_CLIENT_SECRET', // Replace with actual Client Secret  
        businessKey: 'YOUR_BUSINESS_KEY', // Replace with actual Business Key
        sandbox: true // Set to false for production
    };

    const monCash = new MonCashPayment(moncashConfig);

    // Show loading state
    const checkoutBtn = document.querySelector('.checkout-btn');
    const originalText = checkoutBtn.textContent;
    checkoutBtn.textContent = 'Traitement en cours...';
    checkoutBtn.disabled = true;

    // Create payment
    monCash.createPayment(orderData)
        .then(paymentResult => {
            console.log('Payment created:', paymentResult);
            
            // Store order data for later verification
            localStorage.setItem('pending_order', JSON.stringify({
                ...orderData,
                paymentId: paymentResult.payment_id,
                transactionId: paymentResult.transaction_id
            }));
            
            // User will be redirected to MonCash
        })
        .catch(error => {
            console.error('MonCash payment error:', error);
            
            // Restore button state
            checkoutBtn.textContent = originalText;
            checkoutBtn.disabled = false;
            
            // Show error message
            alert('Erreur lors du paiement MonCash. Veuillez réessayer ou choisir le paiement à la livraison.');
            
            // Fallback to cash payment
            if (confirm('Voulez-vous continuer avec le paiement à la livraison?')) {
                orderData.paymentMethod = 'cash';
                processCashOrder(orderData);
            }
        });
}

// Handle MonCash return/callback
function handleMoncashReturn() {
    const urlParams = new URLSearchParams(window.location.search);
    const transactionId = urlParams.get('transactionId');
    const status = urlParams.get('status');
    
    if (transactionId) {
        const pendingOrder = JSON.parse(localStorage.getItem('pending_order') || '{}');
        
        if (status === 'success') {
            // Payment successful
            showPaymentSuccess(pendingOrder);
            
            // Send order to restaurant
            sendOrderToRestaurant(pendingOrder);
            
            // Clear pending order
            localStorage.removeItem('pending_order');
            
        } else if (status === 'failed' || status === 'cancelled') {
            // Payment failed or cancelled
            showPaymentFailure();
            
            // Restore cart
            if (pendingOrder.items) {
                cart = pendingOrder.items;
                updateCartDisplay();
            }
        }
    }
}

// Show payment success message
function showPaymentSuccess(orderData) {
    const successMessage = `
        <div style="text-align: center; padding: 20px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; margin: 20px 0;">
            <h3 style="color: #155724;">✅ Paiement Réussi!</h3>
            <p>Votre commande a été payée avec succès via MonCash.</p>
            <p><strong>Référence:</strong> ${orderData.transactionId}</p>
            <p><strong>Total:</strong> ${orderData.total.toLocaleString()} Gds</p>
            <p>Nous vous contacterons bientôt pour confirmer la livraison.</p>
        </div>
    `;
    
    document.body.insertAdjacentHTML('afterbegin', successMessage);
}

// Show payment failure message
function showPaymentFailure() {
    const failureMessage = `
        <div style="text-align: center; padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; margin: 20px 0;">
            <h3 style="color: #721c24;">❌ Paiement Échoué</h3>
            <p>Le paiement MonCash n'a pas pu être traité.</p>
            <p>Vos articles sont toujours dans votre panier.</p>
            <p>Vous pouvez réessayer ou choisir le paiement à la livraison.</p>
        </div>
    `;
    
    document.body.insertAdjacentHTML('afterbegin', failureMessage);
}

// Send order to restaurant (via WhatsApp, email, or API)
function sendOrderToRestaurant(orderData) {
    const message = `🔔 NOUVELLE COMMANDE PAYÉE - MonCash\n\n` +
        `💳 Paiement: CONFIRMÉ via MonCash\n` +
        `📧 Référence: ${orderData.transactionId}\n\n` +
        `👤 Client: ${orderData.customerName}\n` +
        `📱 Téléphone: ${orderData.customerPhone}\n` +
        `📍 Adresse: ${orderData.deliveryAddress}\n\n` +
        `🍽️ Commande:\n${orderData.items.map(item => 
            `• ${item.name}${item.selectedOption ? ` (${item.selectedOption})` : ''} x${item.quantity} = ${(item.price * item.quantity).toLocaleString()} Gds`
        ).join('\n')}\n\n` +
        `💰 Total: ${orderData.total.toLocaleString()} Gds (PAYÉ)\n` +
        `📝 Instructions: ${orderData.specialInstructions || 'Aucune'}\n\n` +
        `⏰ Commande reçue: ${new Date().toLocaleString('fr-FR')}`;
    
    // Send to restaurant WhatsApp
    const whatsappUrl = `https://wa.me/50937525134?text=${encodeURIComponent(message)}`;
    
    // Also send confirmation to customer
    const customerMessage = `Merci ${orderData.customerName}! 🙏\n\n` +
        `Votre commande chez Lyly's Bar et Resto a été confirmée et payée.\n\n` +
        `📧 Référence: ${orderData.transactionId}\n` +
        `💰 Total payé: ${orderData.total.toLocaleString()} Gds\n\n` +
        `Nous vous contacterons bientôt pour organiser la livraison.\n\n` +
        `Merci de votre confiance! 🍽️`;
    
    const customerWhatsappUrl = `https://wa.me/${orderData.customerPhone.replace(/[^0-9]/g, '')}?text=${encodeURIComponent(customerMessage)}`;
    
    // Open restaurant notification
    window.open(whatsappUrl, '_blank');
    
    // Optionally open customer confirmation (uncomment if needed)
    // setTimeout(() => window.open(customerWhatsappUrl, '_blank'), 2000);
}

// Initialize MonCash return handler when page loads
document.addEventListener('DOMContentLoaded', function() {
    handleMoncashReturn();
});

// Export functions for use in main HTML file
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MonCashPayment,
        enhancedProcessMoncashPayment,
        handleMoncashReturn
    };
}